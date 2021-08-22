#include <bits/stdc++.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int main(){
    Net net = readNet("jasper.onnx");
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);

    int sz[] = {1, 64, 128};
    Mat input(3, &sz[0], CV_32F);
    net.setInput(input);
    Mat out = net.forward();
    std::cout << "out " << out.size << std::endl;
    return 0;
}
