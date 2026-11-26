#!/bin/bash
# Inside the toolchain container (repo mounted at /workspace/mp): turn the downloaded ONNX into something Acuity accepts.
set -e
cd /workspace/mp/vit
python3 ../scripts/vit_prep.py                                   # fix batch=1 / 224x224 -> deit_s1.onnx
python3 -m onnxsim deit_s1.onnx deit_sim.onnx
python3 ../scripts/vit_fold.py                                   # fold the remaining Shape->Gather->Concat chains -> deit_f.onnx
python3 -m onnxsim deit_f.onnx deit_sim2.onnx
python3 ../scripts/vit_erf.py                                    # GELU's Erf -> tanh approximation -> deit_npu.onnx (prints the error check)
