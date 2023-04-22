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

int nc;

int main()
{
    // Load class list.
    std::vector<std::string> class_list;
    // std::ifstream ifs("../models/coco.names");
    std::ifstream ifs("../models/number_plate.names");
    if (!ifs) {
      std::cerr << "Class names file not found! Please check the path relative to the pwd" << std::endl;
      return -1;
    }

    
    std::string line;
    
    
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    nc = class_list.size();
    std::cout << "Number of classes: " << nc << std::endl;
    // Load image.
    cv::Mat frame;
    // frame = cv::imread("sample.jpg");
    frame = cv::imread("MicrosoftTeams-image.png");
    // Load model.
    cv::dnn::Net net;
    try {
        // net = cv::dnn::readNet("../models/yolov5s.onnx");
        net = cv::dnn::readNet("../models/best_include_torchscript.onnx");
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        std::cerr << "Please check the path relative to the pwd" << std::endl;

        // Handle the error
        // ...
        return -1;
    }

    std::vector<cv::Mat> detections;     // Process the image.
    detections = pre_process(frame, net);
    cv::Mat frame_cloned = frame.clone();
    std::vector<cv::Rect> boxes_NMS;
    std::vector<std::string> labels_NMS;
    
    cv::Mat img = post_process(frame_cloned, detections, class_list, boxes_NMS, labels_NMS);
    for (int i = 0; i < boxes_NMS.size(); i++)
    {
        // Draw bounding box.
        cv::rectangle(img, cv::Point(boxes_NMS[i].x, boxes_NMS[i].y), cv::Point(boxes_NMS[i].x + boxes_NMS[i].width, boxes_NMS[i].y + boxes_NMS[i].height), BLUE, 3*THICKNESS);
        // Draw class labels.
        draw_label(img, labels_NMS[i], boxes_NMS[i].x, boxes_NMS[i].y);
        std::cout << "x: " <<  boxes_NMS[i].x << std::endl;
        std::cout << "y: " <<  boxes_NMS[i].y << std::endl;
        std::cout << "width: " <<  boxes_NMS[i].width << std::endl;
        std::cout << "height: " <<  boxes_NMS[i].height << std::endl;
        std::cout << "Frame size: " << frame.size << std::endl;
        std::cout << "y range " << boxes_NMS[i].y + boxes_NMS[i].height << std::endl;
        std::cout << "x range " << boxes_NMS[i].x + boxes_NMS[i].width << std::endl;



        // cv::Mat plate_img = frame(cv::Range(100 , 150), cv::Range(200,300));
        
        // Feed the bounding box information to tesseract to do OCR
        cv::Rect roi(boxes_NMS[i].x, boxes_NMS[i].y, boxes_NMS[i].width, boxes_NMS[i].height);
        cv::Mat plate_img = frame(roi);
        cv::imshow("plate img", plate_img);
        cv::waitKey(0);
        std::cout << "Performing OCR" << std::endl;
        std::string ocr_text;
        
        cv::Mat test_img = cv::imread("ma.png");
        
        ocrTOtext(plate_img,ocr_text);
        std::cout << "Detected text is: " << ocr_text << std::endl;
    }

    
    
    



    // Put efficiency information.
    // The function getPerfProfile returns the overall time for     inference(t) and the timings for each of the layers(in layersTimes).
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time : %.2f ms", t);
    // std::string label = "Inference time";
    cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    cv::imshow("Output", img);
    cv::waitKey(0);
    return 0;
}