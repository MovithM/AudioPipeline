import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from scipy.signal import resample


import numpy as np
import sounddevice as sd
import queue
import time

from faster_whisper import WhisperModel


class FastWhisperNode(Node):
    def __init__(self):
        super().__init__('fastwhisper_node')

        self.publisher_ = self.create_publisher(String, '/speech_text', 10)
        self.subscription = self.create_subscription(
            Bool,
            '/wake_word',
            self.wakeword_callback,
            10
        )

        self.model = WhisperModel("small", device="cpu", compute_type="int8")

        self.sample_rate = 48000
        self.block_size = 1024
        self.audio_queue = queue.Queue()
        self.recording = False

        self.get_logger().info("FastWhisper Live STT Node Ready")

    # ------------------------
    def wakeword_callback(self, msg):
        if msg.data and not self.recording:
            self.get_logger().info("Wakeword detected → Start listening")
            self.recording = True
            time.sleep(0.05)   # small sync delay
            self.listen_and_transcribe()


    # ------------------------
    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_queue.put(indata.copy())
    
    # ------------------------
    def listen_and_transcribe(self):


        audio_frames = []

        # Clear old audio
        while not self.audio_queue.empty():
            self.audio_queue.get()

        time.sleep(0.2)  # mic warm-up

        silence_threshold = 0.002
        silence_duration = 1.0
        max_duration = 8.0

        start_time = time.time()
        last_voice_time = time.time()

        #  START MICROPHONE STREAM
        with sd.InputStream(
            samplerate=self.sample_rate,
            dtype='float32',
            blocksize=self.block_size,
            callback=self.audio_callback,
            device="default"
        ):



            while True:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_frames.append(chunk)

                    rms = np.sqrt(np.mean(chunk**2))

                    if rms > silence_threshold:
                        last_voice_time = time.time()

                if time.time() - last_voice_time > silence_duration:
                    break

                if time.time() - start_time > max_duration:
                    break

        # STOP RECORDING
        self.recording = False

        if len(audio_frames) == 0:
            self.get_logger().warn("No audio captured")
            return

        audio = np.concatenate(audio_frames, axis=0)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # resample to 16k
        audio = resample(audio, int(len(audio) * 16000 / 48000))




        segments, info = self.model.transcribe(
            audio,
            language="en"
        )




        text = " ".join([seg.text for seg in segments])

        msg = String()
        msg.data = text.strip()

        self.publisher_.publish(msg)
        self.get_logger().info(f"Published: {msg.data}")



def main():
    rclpy.init()
    node = FastWhisperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
