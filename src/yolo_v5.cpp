/*Dev Vaibhav, Siddharth Maheshwari, Skanda Akkihebbal Prasanna
Spring 2023 CS 5330
Final Project: Autonomous Lane and Number plate detection using classic Computer Vision and Deep Learning 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h> // To use sleep functionality
#include <helper_functions.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

int camera_ID = 1; // Camera ID for the webcam

// Resolution required for the window
int res_width = 600; //columns
int res_height = res_width*9/16; //rows

// Main function which captures frames from the image/ video (provided as input argument to CLI)/ live feed (no argument)
int main(int argc, char** argv)
{
    //Source: 0 :image | 1 : video from file | 2 : live video from webcam
    
    char target_filename[256];
    char *target_filename_char_star;
    int source = 0; //0 :image | 1 : video from file | 2 : live video from webcam

    cv::Mat lane_detected; // Matrix which stores the detected output

    // If CLI argument is provided, extract image/ video file name from this
    if( argc == 2) {
        strcpy(target_filename, argv[1] );
        target_filename_char_star = target_filename;
        std::cout << "Reading image/ video from directory" << std::endl;
        source = 0; //It could be 0 (image) or 1 (video)
    }else{
        source = 2; // Live feed
    }

    cv::Mat frame; // Matrix which stores the frame to process
    cv::VideoCapture *capdev; // Capture device to read the image(s)
    int fps; // Stores fps at which the video was recorded/ is being streamed

    int window_id = 1; // Used as part of a label to display the output
    
    cv::String window_original_image = std::to_string(window_id) + " :Original image";
    cv::namedWindow(window_original_image);
    window_id++;

    cv::String window_lanes_detected = std::to_string(window_id) + " :Lanes detected";
    cv::namedWindow(window_lanes_detected);
    window_id++;

    cv::Size refS;
    // Decide if input source is image or video file by calculating total number of frames in input
    if(source == 0){
        // Check if the input is image or video file by counting the number of frames in the input
        
        capdev = new cv::VideoCapture(target_filename_char_star); 
        // Print error message if the stream is invalid
        if (!capdev->isOpened()){
            std::cout << "Error opening video stream or image file" << std::endl;
        }else{
            // Obtain fps and frame count by get() method and print
            // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
            fps = capdev->get(5);
            std::cout << "Frames per second in video:" << fps << std::endl;

            refS = cv::Size( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                    (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
            printf("Image size(WidthxHeight) from file: %dx%d\n", refS.width, refS.height);
        
            // Obtain frame_count using opencv built in frame count reading method
            // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
            int frame_count = capdev->get(7);
            
            // PNG photos return negative frame_count. This check handles this
            if (frame_count < 0){
                frame_count = 1;
            }
            std::cout << "  Frame count :" << frame_count << std::endl;

            (frame_count == 1) ? source = 0 : source = 1;
            std::cout << "Source is: " << source << std::endl;
        }

        if (source == 0){
            // Read the image file
            frame = cv::imread(target_filename_char_star,cv::ImreadModes::IMREAD_COLOR);
            std::cout << "Reading image from disk successful. Number of channels in image: " << frame.channels() << std::endl;
            
            // Check for failure
            if (frame.empty()) {
                std::cout << "Could not open or find the image" << std::endl;
                // std::cin.get(); //wait for any key press
                return -1;
            }
        }
    }

    
    // Source is live video
    if (source == 2){
        capdev = new cv::VideoCapture(camera_ID);
        
        if (!capdev->isOpened()) {
            throw std::runtime_error("Error");
            return -1;
        }
        fps = capdev->get(5);
        std::cout << "Input feed is camera" << std::endl;
        // get some properties of the image
        refS = cv::Size( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                    (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
    }


    // LOAD THE OBJECT DETECTION MODEL: YOLOV5

    // Load class list.
    std::vector<std::string> class_list_1; // Stores 80 classes of yolov5s
    std::vector<std::string> class_list_2; // Stores one class of custom trained model, number plate
    std::ifstream ifs_1("../models/coco.names");
    std::ifstream ifs_2("../models/number_plate.names");
    if (!ifs_1 || !ifs_2) {
      std::cerr << "Class names file not found! Please check the path relative to the pwd" << std::endl;
      return -1;
    }    
    std::string line;

    while (getline(ifs_1, line)){
        class_list_1.push_back(line);
    }

    while (getline(ifs_2, line)){
        class_list_2.push_back(line);
    }

    int nc_1 = class_list_1.size(); // Number of classes for first model
    int nc_2 = class_list_2.size(); // Number of classes for second model
    std::cout << "Number of classes: " << nc_1 << " | " << nc_2 << std::endl;

    // Load model.
    cv::dnn::Net net_1;
    cv::dnn::Net net_2;
    try {
        net_1 = cv::dnn::readNet("../models/yolov5s.onnx");
        net_2 = cv::dnn::readNet("../models/best_include_torchscript.onnx");
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        std::cerr << "Please check the path relative to the pwd" << std::endl;
        return -1;
    }

    cv::VideoWriter video_original("saved_video_original.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(refS.width,refS.height));

    int key_pressed; // Stores the key pressed by the user when the function is running
    int save_video = 0;
    int select_polygon = 1;
    while (true) {
        // std::cout << "####################### REACHED START OF WHILE LOOP #######################" << std::endl;
        // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        // std::cout << "Getting data from camera" << std::endl;
        
        // Read video file as source frame if source == 1
        if( source == 1){
            bool isSuccess = capdev->read(frame);

            // If video file has ended playing, play it again (i.e. in loop)
            if (isSuccess == false){
                std::cout << "Video file has ended. Running the video in loop" << std::endl;
                capdev->set(1,0); // Read the video file from beginning | https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704da6223452891755166a4fd5173ea257068
                capdev->read(frame); //Src: https://stackoverflow.com/questions/17158602/playback-loop-option-in-opencv-videos
            }
        }else if( source == 2){
            // std::cout << "Getting data from camera" << std::endl;
            *capdev >> frame; //frame.type() is 16 viz. 8UC3
        }

        // Resize the image to make computation easier
        // cv::resize(frame, frame, cv::Size(res_width, res_height));
        // See the original frame
        cv::imshow(window_original_image, frame);


        // Perform object detection using YOLOv5 pre-trained model
        detect_objects(frame, nc_1, class_list_1, net_1, lane_detected);

        // Perform license plate detection and OCR using custom trained model
        detect_objects(lane_detected, nc_2, class_list_2, net_2, lane_detected);
        
        // Perform lane segmentation using classical CV
        
        
        // Ask user to select the polygon only once
        cv::Mat mask;
        
        if (select_polygon == 1){
            struct MouseCallbackData{
                cv::Mat frame;
                std::vector<cv::Point> vertices;
                };
        
            std::vector<cv::Point> vertices;
            MouseCallbackData data{frame, vertices}; // Initializing the structure

            // Display the sixth frame and wait for the user to select the ROI
            cv::namedWindow("Select ROI");
            
            std::cout << "Started mouse callback function" << std::endl;
            cv::imshow("Select ROI", data.frame);
            cv::setMouseCallback("Select ROI", selectROI, &data);
            // cv::waitKey(0);
            std::cout << "Ended mouse callback function" << std::endl;
            
            while (data.vertices.size() < 4) {
                cv::waitKey(1);
            }
        
            std::cout << "Vertices size is: " << data.vertices.size() << std::endl;
            // Create the mask region that corresponds to the ROI
            // cv::Mat mask = data->mask;
            vertices = data.vertices;
            mask = createMask(frame, vertices);
            select_polygon = 0;
        }
        
        lane_detection(lane_detected, lane_detected, mask);

        // Show the original image with lanes and detected objects
        cv::imshow(window_lanes_detected, lane_detected);

        if(source == 0){
            // If source frame is image, don't run the processing again and again. So, wait indefinitely for user's input
            key_pressed = cv::waitKey(0);
        }else{
            key_pressed = cv::waitKey(1000/fps); // It will play video at its original fps
        }
        
        if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
            //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
            std::cout << "q is pressed. Exiting the program" << std::endl;
            cv::destroyWindow("1: Original_Image"); //destroy the created window
            video_original.release();
            return 0;
        }

        if (key_pressed == 52){//Let the user save the original video when 4 is pressed
            save_video = 1;
            std::cout << "Saving original video!" << std::endl;
        }

        if (save_video == 1){
            video_original.write(frame);
        }
    }
 
return 0;
}