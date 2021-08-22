import sys
import cv2 as cv
import numpy as np
import soundfile as sf

net = cv.dnn.readNetFromONNX('jasper.onnx')
input = np.random.randn(2,64,128)
# print(input)
# Maybe add print function before doing anything in setInput
input = np.pad(input, ((0,0),(0,0),(0,max(0,513-input.shape[2]))), 'constant', constant_values=0)
net.setInput(np.array(input))
out = net.forward()
print(out)