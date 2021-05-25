import onnx
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
    model.graph.initializer.insert(i,init)

with open('jasper_dynamic_input_float.onnx','wb') as f:
    onnx.save_model(model,f)