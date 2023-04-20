/*Dev Vaibhav, Siddharth Maheshwari, Skanda Akkihebbal Prasanna
Spring 2023 CS 5330
Final Project: Autonomous Lane and Number plate detection using classic Computer Vision and Deep Learning 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h> // To use sleep functionality
#include <filter.h>
#include <helper_functions.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <modes.h>
#include <fstream>


// Src: https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/
// https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python

// How to build the project and run the executable: https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html
// clear && cmake . && make #The executable gets stored into the bin folder

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5; // To filter low probability class scores.
const float NMS_THRESHOLD = 0.45; // To remove overlapping bounding boxes.
const float CONFIDENCE_THRESHOLD = 0.45; // Filters low probability detections.

const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX; // Its enumeration is zero | Src: https://docs.opencv.org/
const int THICKNESS = 1;

cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0,0,255);

int main()
{
    // Load class list.
    std::vector<std::string> class_list;
    std::ifstream ifs("coco.names");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    // Load image.
    cv::Mat frame;
    frame = cv::imread("traffic.jpg");
    // Load model.
    cv::dnn::Net net;
    net = cv::dnn::readNet("YOLOv5s.onnx");
    std::vector<cv::Mat> detections;     // Process the image.
    detections = pre_process(frame, net);
    cv::Mat frame_cloned = frame.clone();
    cv::Mat img = post_process(frame_cloned, detections, class_list);
    // Put efficiency information.
    // The function getPerfProfile returns the overall time for     inference(t) and the timings for each of the layers(in layersTimes).
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time : %.2f ms", t);
    // std::string label = "Inference time";
    cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    imshow("Output", img);
    cv::waitKey(0);
    return 0;
}