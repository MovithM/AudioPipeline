#!/usr/bin/env python3
import json

import os
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from transformers import DistilBertTokenizerFast
from intent_pkg.coordinate_ext import extract_coordinates


# -------------------------------
# Joint Intent + Slot Model
# -------------------------------
class JointIntentSlotModel(torch.nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        from transformers import DistilBertModel

        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        hidden = self.distilbert.config.hidden_size

        self.intent_classifier = torch.nn.Linear(hidden, num_intents)
        self.slot_classifier = torch.nn.Linear(hidden, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits


# -------------------------------
# ROS2 Intent Node
# -------------------------------
class IntentNode(Node):
    def __init__(self):
        super().__init__("intent_node")

        # 🔒 ABSOLUTE PATHS (important for ROS install)
        home = os.path.expanduser("~")
        model_path = os.path.join(
            home, "ros2_ws/src/intent_pkg/models/intent_model.pt"
        )
        tokenizer_path = os.path.join(
            home, "ros2_ws/src/intent_pkg/models/intent_tokenizer"
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

 
        self.intent_labels = [
            "INT-OBSERVE",
            "INT-NAV",
            "INT-SEARCH",
            "INT-STOP",
            "INT-IDLE"
        ]

        self.slot_labels = [
            "O",
            "B-OBJECT",
            "I-OBJECT",
            "B-LOCATION",
            "I-LOCATION"
        ]


        # Load tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            tokenizer_path
        )

        # Load model
        self.model = JointIntentSlotModel(
            num_intents=len(self.intent_labels),
            num_slots=len(self.slot_labels)
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        # ROS interfaces
        self.create_subscription(
            String,
            "/speech_text",
            self.callback,
            10
        )

        self.intent_pub = self.create_publisher(
            String,
            "/intent",
            10
        )

        self.slot_pub = self.create_publisher(
            String,
            "/slots",
            10
        )

        self.get_logger().info("Intent node ready")

    # -------------------------------
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            intent_logits, slot_logits = self.model(**inputs)

        intent_id = torch.argmax(intent_logits, dim=1).item()
        slot_ids = torch.argmax(slot_logits, dim=2)[0]

        tokens = self.tokenizer.tokenize(text)
        slots = [self.slot_labels[i] for i in slot_ids[:len(tokens)]]

        return self.intent_labels[intent_id], list(zip(tokens, slots))

    # -------------------------------

    def callback(self, msg):
        text = msg.data

        intent, slots = self.predict(text)
        x, y = extract_coordinates(text)

        payload = {
            "intent": intent,
            "x": x,
            "y": y,
            "slots": slots
        }

        self.intent_pub.publish(
            String(data=json.dumps(payload))
        )

        self.get_logger().info(json.dumps(payload))





# -------------------------------
def main():
    rclpy.init()
    node = IntentNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
