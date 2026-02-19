# AudioPipeline
This repo contains the programs and systerm requirements for the audio pipeline wrt wenee the pet robot
# File Architecture
(Kindly install ros humble to connect the nodes)
WakeWord_pkg
|    |_ Wakeword_node.py (Pretrained onnx file used - wenee_50k.onnx)
|
Fastwhisper_pkg
|    |_ FastWhisper_node.py (Fastwhisper small (directly installed from the sourcepage))
|
Intent_pkg
     |_ Coordinate_ext.py
     |_ Intent_node.py (Pretrained model used - Intent_model (large file (approx 250mb) kindly mail movithm02@gmail.com))
                       (Tokenizer used        - Intent_tokenizer)
