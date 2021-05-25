import cv2 as cv
import numpy as np

print("before calling")
net = cv.dnn.readNetFromONNX('./jasper_input_1x64x256_float.onnx') # N,C,W
# net1 = cv.dnn.readNetFromONNX('./jasper_dynamic_input_float.onnx')

