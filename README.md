# AudioPipeline
This repo contains the programs and systerm requirements for the audio pipeline wrt wenee the pet robot

# File Architecture
(Kindly install ros humble to connect the nodes)
WakeWord_pkg -> Wakeword_node.py (Pretrained onnx file used - wenee_50k.onnx)
Fastwhisper_pkg -> FastWhisper_node.py (Fastwhisper small (directly installed from the sourcepage))
Intent_pkg ->  Coordinate_ext.py,Intent_node.py (Pretrained model used - Intent_model (large file (approx 250mb) kindly mail movithm02@gmail.com))(Tokenizer used - Intent_tokenizer)

## Dependencies
(All nodes run inside a Python virtual environment.)
Full dependency list is available in `requirements.txt`.
# Wakeword Detection
- openwakeword
- onnxruntime
- numpy
- sounddevice
- scipy
- rclpy
# Speech-to-Text (FastWhisper)
- faster-whisper
- ctranslate2
- torch
- librosa
- sentencepiece
- numpy
- sounddevice
- scipy
- rclpy
# Intent & Slot Filling
- torch
- transformers
- tokenizers- scikit-learn
- numpy
- rclpy
