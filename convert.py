import onnx
from onnx import numpy_helper
import numpy as np

# Load base onnx model
model = onnx.load("./jasper-onnx/1/model.onnx")

# Change input data type FLOAT16 ==> FLOAT
inp = model.graph.input[0]
model.graph.input.remove(inp)
inp.type.tensor_type.elem_type = 1
model.graph.input.insert(0,inp)

# Change output data type FLOAT16 ==> FLOAT
out = model.graph.output[0]
model.graph.output.remove(out)
out.type.tensor_type.elem_type = 1
model.graph.output.insert(0,out)

# Change data type FLOAT16 ==> FLOAT of every initilizer
for i,init in enumerate(model.graph.initializer):
    model.graph.initializer.remove(init)
    init.data_type = 1
    init.raw_data = np.frombuffer(init.raw_data, count=np.product(init.dims), dtype=np.float16).astype(np.float32).tobytes()
    model.graph.initializer.insert(i,init)
    
# This part adds an additional reshape node to handle the inconsistant input
# from python and c++ of openCV. see https://github.com/opencv/opencv/issues/19091
# 1. Make & insert a new node with 'Reshape' operation & required initializer
tensor = numpy_helper.from_array(np.array([0,64,-1]),name='shape_reshape')
model.graph.initializer.insert(0,tensor)
node = onnx.helper.make_node(op_type='Reshape',inputs=['input__0','shape_reshape'], outputs=['input_reshaped'], name='reshape__0')
model.graph.node.insert(0,node)
# 2. Fix connection on next node
model.graph.node[1].input[0] = 'input_reshaped'

# Save FP32 model
with open('jasper_dynamic_input_float.onnx','wb') as f:
    onnx.save_model(model,f)