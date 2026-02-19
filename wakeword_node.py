import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

import sounddevice as sd
import numpy as np
from openwakeword.model import Model

MODEL_PATH = "/home/movithm/ros2_ws/models/weenee_50k.onnx"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
THRESHOLD = 0.25

class WakeWordNode(Node):
    def __init__(self):
        super().__init__("wakeword_node")

        self.publisher = self.create_publisher(Bool, "/wake_word", 10)

        self.model = Model(
            wakeword_models=[MODEL_PATH],
            inference_framework="onnx"
        )

        self.get_logger().info("Wakeword(weenee) loaded")

        self.hit_count = 0

        self.listen_loop()

    def listen_loop(self):
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE
        ) as stream:

            self.get_logger().info("Say: weenee")

            while rclpy.ok():
                data, _ = stream.read(CHUNK_SIZE)
                audio = np.frombuffer(data, dtype=np.int16)

                prediction = self.model.predict(audio)
                score = list(prediction.values())[0]

                if score > THRESHOLD:
                    self.hit_count += 1
                else:
                    self.hit_count = 0

                if self.hit_count >= 1:
                    msg = Bool()
                    msg.data = True
                    self.publisher.publish(msg)

                    self.get_logger().info(f"Wakeword detected (score={score:.2f})")

                    self.hit_count = 0
                    self.model.reset()

def main():
    rclpy.init()
    node = WakeWordNode()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
