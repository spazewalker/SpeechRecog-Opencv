#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

int main(int argc, char** argv)
{
    String model = "./jasper_dynamic_input_float.onnx";
    Net net = cv::dnn::readNetFromONNX(model);

    return 0;
}



// Output
// [DEBUG:0] global /home/shivanshu/opencv/modules/dnn/src/onnx/onnx_importer.cpp (73) ONNXImporter DNN/ONNX: processing ONNX model from file: ./jasper_input_1x64x256_float.onnx
// [ INFO:0] global /home/shivanshu/opencv/modules/dnn/src/onnx/onnx_importer.cpp (422) populateNet DNN/ONNX: loading ONNX v6 model produced by 'pytorch':1.7. Number of nodes = 327, inputs = 1, outputs = 1
// [DEBUG:0] global /home/shivanshu/opencv/modules/dnn/src/onnx/onnx_importer.cpp (434) populateNet DNN/ONNX: graph simplified to 327 nodes
// Segmentation fault (core dumped)