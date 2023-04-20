/*Dev Vaibhav
Spring 2023 CS 5330
Project 4: Calibration and Augmented Reality
*/

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "helper_functions.h"
#include "filter.h"

// Do nothing when the user changes the threshold on GUI trackbar
static void on_trackbar( int, void* ){
    //It does nothing
}    


int main( int argc, char** argv ){
    cv::Mat frame, frame_gray;
    cv::VideoCapture *capdev;

    capdev = new cv::VideoCapture(1);
    if (!capdev->isOpened()) {
        throw std::runtime_error("Error");
        return -1;
    }
    std::cout << "Input feed is camera" << std::endl;
    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
 
    int window_id = 1;
    cv::String window_original_image = std::to_string(window_id) + " :Original image";
    cv::namedWindow(window_original_image);
    window_id++;

    cv::String window_harris_corners = std::to_string(window_id) + " :Harris Corner Detector";
    cv::namedWindow(window_harris_corners);
    window_id++;

    int thresh = 200;
    int MAX_THRESH = 255;
    
    std::cout << " ############### REACHED HERE ###############" << std::endl;
    cv::createTrackbar( "Threshold: ", window_harris_corners, &thresh, MAX_THRESH, on_trackbar );
    
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::Mat dst = cv::Mat::zeros( frame.size(), CV_32FC1 );
    cv::Mat dst_norm, dst_norm_scaled;

    while (true) {
        // std::cout << "####################### REACHED START OF WHILE LOOP #######################" << std::endl;
        // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
        // std::cout << "Getting data from camera" << std::endl;
        *capdev >> frame; //frame.type() is 16 viz. 8UC3

        cv::resize(frame, frame, cv::Size(res_width, res_height));    
        // printf("Resizing image to %dx%d\n", res_width, res_height);
        
        int key_pressed = cv::waitKey(1); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
        // std::cout << "Pressed key is: " << key_pressed << std::endl;

        // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
        if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
            //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
            std::cout << "q is pressed. Exiting the program" << std::endl;
            cv::destroyWindow("1: Original_Image"); //destroy the created window
            return 0;
        }
        
        greyscale(frame, frame_gray);
        // std::cout << "Gray image is: " << frame_gray << std::endl;
        // Src: https://anothertechs.com/programming/cpp/opencv-corner-harris-cpp/
        cv::cornerHarris( frame_gray, dst, blockSize, apertureSize, k );
        cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
        cv::convertScaleAbs( dst_norm, dst_norm_scaled );
        for( int i = 0; i < frame_gray.rows ; i++ ){
            for( int j = 0; j < frame_gray.cols; j++ ){
                if( (int) dst_norm.at<float>(i,j) > thresh ){
                    cv::circle( dst_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(255), 2, 8, 0 );
                }
            }
        }            

        
        cv::imshow( window_original_image, frame );
        cv::imshow(window_harris_corners, dst_norm_scaled);
    }
    return 0;
}