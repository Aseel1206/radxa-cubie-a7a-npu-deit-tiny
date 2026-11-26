import onnx, numpy as np, sys
from onnx import numpy_helper, helper, TensorProto
m=onnx.load('/workspace/mp/vit/deit_tiny.onnx')
for o in m.graph.output:
    d=o.type.tensor_type.shape.dim; d[0].dim_value=1
m.graph.input[0].type.tensor_type.shape.dim[0].dim_value=1
for i,k in zip(m.graph.input[0].type.tensor_type.shape.dim[1:],(3,224,224)): i.dim_value=k
onnx.save(m,'/workspace/mp/vit/deit_s1.onnx')
