#!/bin/bash
# Download the ONNX export of facebook/deit-tiny-distilled-patch16-224 (Xenova's transformers.js export).
set -e; mkdir -p vit; cd vit
B=https://huggingface.co/Xenova/deit-tiny-distilled-patch16-224/resolve/main
curl -sSL -o deit_tiny.onnx $B/onnx/model.onnx; curl -sSL -o config.json $B/config.json; ls -l
