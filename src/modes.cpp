/*Dev Vaibhav
Spring 2023 CS 5330
Project 4: Calibration and Augmented Reality
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h> // To use sleep functionality
#include "helper_functions.h"
#include "filter.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <csv_util.h>

// Import the aruco module in OpenCV
#include <opencv2/aruco.hpp>
 




// Calibrates the camera by finding its intrinsic and extrinsic parameters
int calibrate_camera(){

  char dirname[256] = "../calibration_images/";
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;

  // Src: https://stackoverflow.com/questions/2802188/file-count-in-a-directory-using-c
  // get the directory path
  // strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );
  int i = 0;
  // open the directory
  dirp = opendir( dirname );
  if( dirp != NULL) {
    while (dp = readdir (dirp)){
      // check if the file is an image
      if( strstr(dp->d_name, ".jpg") ||
          strstr(dp->d_name, ".jpeg") ||
          strstr(dp->d_name, ".png") ||
          strstr(dp->d_name, ".ppm") ||
          strstr(dp->d_name, ".tif") ){
        i++;
          }
    }
    (void) closedir (dirp);    
  }else{
    perror ("Couldn't open the directory");
    exit(-1);
  }
  printf("There's %d files in the current directory.\n", i);

  if (i == 0){
    //Capture calibration images from the camera if the directory is empty
    capture_calibration_images();  
  }

  //READ all the images from a directory, calculate calibration parameters using them and writes them to a yaml file
  calc_calib_params(dirname);

 return 0;
}

// Project a virtual object to the scene if checkerboard is detected
int project(int &source, char *target_filename_char_star, int &alter_base){

  //Source: 0 :image | 1 : video from file | 2 : live video from webcam
  cv::Mat frame;
  cv::VideoCapture *capdev;
  int fps;
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
  
      // Obtain frame_count using opencv built in frame count reading method
      // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
      int frame_count = capdev->get(7);
      std::cout << "  Frame count :" << frame_count << std::endl;

      (frame_count == 1) ? source = 0 : source = 1;
      std::cout << "Source is: " << source << std::endl;
    }

    if (source == 0){
      // Read the image file
      frame = cv::imread(target_filename_char_star);//,cv::ImreadModes::IMREAD_UNCHANGED);
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
    capdev = new cv::VideoCapture(1);
    if (!capdev->isOpened()) {
      throw std::runtime_error("Error");
      return -1;
    }
    fps = capdev->get(5);
    std::cout << "Input feed is camera" << std::endl;
    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
  }
    

  // Read calibration parameters from the yaml file (calibparam.yaml)
  std::cout << std::endl << "Reading: " << std::endl;
  cv::FileStorage fs;
  fs.open("calibparam.yaml", cv::FileStorage::READ);
  int itNr;
  //fs["iterationNr"] >> itNr;
  itNr = (int) fs["iterationNr"];
  std::cout << itNr << std::endl;
  if (!fs.isOpened())
  {
      std::cerr << "Failed to open " << "calibparam.yaml" << std::endl;
      return 1;
  }
  
  cv::FileNode n;// = fs["strings"];  
                         // Read string sequence - Get node
  // if (n.type() != cv::FileNode::SEQ)
  // {
  //     std::cerr << "strings is not a sequence! FAIL" << std::endl;
  //     return 1;
  // }
  // cv::FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
  // for (; it != it_end; ++it)
  //     std::cout << (std::string)*it << std::endl;
  
  // n = fs["Mapping"];                                // Read mappings from a sequence
  // std::cout << "Two  " << (int)(n["Two"]) << "; ";
  // std::cout << "One  " << (int)(n["One"]) << std::endl << std::endl;
  // MyData m;
  // fs["MyData"] >> m;                                 // Read your own structure_
  // std::cout << "MyData = " << std::endl << m << std::endl << std::endl;

  cv::Mat cameraMatrix;
  cv::Vec<float, 5> distCoeffs;
  std::vector<std::vector<cv::Vec3f>> point_list;

  fs["Camera_Matrix"] >> cameraMatrix;                                      // Read cv::Mat
  fs["Distortion_Coefficients"] >> distCoeffs;
  fs["Point_List"] >> point_list;


  std::cout << std::endl
      << "Camera_Matrix = " << cameraMatrix << std::endl;
  // std::cout << " ############### REACHED HERE ############### " << std::endl;

  std::cout << "Distortion_Coefficients = " << distCoeffs << std::endl << std::endl;
  std::cout << "Point_List = " << point_list[0][0] << std::endl << std::endl;
  //Show default behavior for non existing nodes
  // std::cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
  // fs["NonExisting"] >> m;
  // cout << endl << "NonExisting = " << endl << m << endl;
  
  std::cout << std::endl
  << "Tip: Open up " << "calibparam.yaml" << " with a text editor to see the serialized data." << std::endl;
  fs.release();

  // Start a video loop

  // CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
  // For instance, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels. There are types predefined for up to four channels. https://docs.opencv.org/3.4/d6/d6d/tutorial_mat_the_basic_image_container.html
  
    
  int window_id = 1;
  cv::String window_original_image = std::to_string(window_id) + " :Original image";
  window_id++;

  cv::Mat grey_image;
  
  std::vector<cv::Point2f> corner_set; //this will be filled by the detected corners
  bool patternfound; // Its true when all 54 corners are detected in the checkerboard
  cv::Mat rvec, tvec;

  int key_pressed; // Stores the key pressed by the user when the function is running

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

    cv::resize(frame, frame, cv::Size(res_width, res_height));    
    // printf("Resizing image to %dx%d\n", res_width, res_height);
    
    
    cv::Mat temp; // Using temp matrix to detected multiple checkerboards
    frame.copyTo(temp);

    // STEP 1: Detect and Extract Chessboard Corners
    // Src: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
    
    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    
    patternfound = false;
    int num_patters_found = 0;
    while(patternfound == true || num_patters_found == 0 ){
      greyscale(temp, grey_image); // Converting the image to greyscale
      patternfound = findChessboardCorners(grey_image, patternsize, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
            + cv::CALIB_CB_FAST_CHECK);
      
      if (patternfound == false){
        // Check on the next frame
        break;
      }
      cornerSubPix(grey_image, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      // drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);
      // std::cout << "Number of corners found: " << corner_set.size() << std::endl;
      
      //Src: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
      cv::solvePnP(point_list[0], corner_set, cameraMatrix, distCoeffs, rvec, tvec );

      // Project a 3D point (0, 0, 1000.0) onto the image plane.
      // We use this to draw a line sticking out of the nose
      // draw_shape_on_image("axis", frame, cameraMatrix, distCoeffs, rvec, tvec, corner_set, alter_base);
      draw_shape_on_image("wa_monument", frame, cameraMatrix, distCoeffs, rvec, tvec, corner_set, alter_base);

      //whiten the pixels for the found checkerboard and find the next
      std::vector<cv::Point> checker_board_ibb; //checkerboard internal bounding box
      checker_board_ibb.push_back(corner_set[0]); //TL
      checker_board_ibb.push_back(corner_set[8]); //TR
      checker_board_ibb.push_back(corner_set[53]); //BR
      checker_board_ibb.push_back(corner_set[45]); //BL

      cv::fillConvexPoly(temp, checker_board_ibb, cv::Scalar(255,255,255) ); //Painting the deected checkerboard's internal corners in white to detect multiple checkerboards| https://stackoverflow.com/questions/51772947/multiple-checkerboard-detection-opencv
      num_patters_found++;
      // cv::imshow("Painted in white" , temp);
    }
    std::cout << "Number of patterns found: " << num_patters_found << std::endl;

  
    cv::imshow(window_original_image, frame);
    //  = cv::waitKey(1); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
    // std::cout << "Pressed key is: " << key_pressed << std::endl;

    // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
    
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
      return 0;
    }
  }
  return 0;
}

// Project an image or recorded video on the source frame if four aruco tags are detected
int arucoTV(int &ar_source, char *target_filename_char_star){

  int test = 0; //Load a test video to check aruco tag detection code. Used for debugging

  cv::Mat frame, frame_ar;
  cv::VideoCapture *capdev, *capdev2; //capdev is for the live feed from webcam | capdev2 is for the ar frame which needs to be superimposed on the Aruco markers
  int fps; //Stores frame per second of the AR image/ video/ live feed
  
  //ar_source: 0 :image | 1 : video from file | 2 : live video from webcam
  if(ar_source == 0){
    // Check if the input is image or video file by counting the number of frames in the input
    
    capdev2 = new cv::VideoCapture(target_filename_char_star); 
    // Print error message if the stream is invalid
    if (!capdev2->isOpened()){
      std::cout << "Error opening video stream or image file" << std::endl;
    }else{
      // Obtain fps and frame count by get() method and print
      // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
      fps = capdev2->get(5);
      std::cout << "Frames per second in video:" << fps << std::endl;
  
      // Obtain frame_count using opencv built in frame count reading method
      // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
      int frame_count = capdev2->get(7);
      std::cout << "  Frame count :" << frame_count << std::endl;

      // If no. of frames is 1, it is an image, otherwise a video
      (frame_count == 1) ? ar_source = 0 : ar_source = 1;   
    }

    if (ar_source == 0){
      // Read the image file
      frame_ar = cv::imread(target_filename_char_star);//,cv::ImreadModes::IMREAD_UNCHANGED);
      std::cout << "Reading image from disk successful. Number of channels in image: " << frame.channels() << std::endl;
      
      // Check for failure
      if (frame_ar.empty()) {
          std::cout << "Could not open or find the image" << std::endl;
          // std::cin.get(); //wait for any key press
          return -1;
      }
    }
  }
  
  // Read live feed from webcam
  capdev = new cv::VideoCapture(1);
  if (!capdev->isOpened()) {
    throw std::runtime_error("Error");
    return -1;
  }
  fps = capdev->get(5);
  std::cout << "Input feed is camera" << std::endl;
  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
              (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Image size(WidthxHeight) from camera: %dx%d\n", refS.width, refS.height);
  
  std::cout << "ar_source is: " << ar_source << std::endl;

  int window_id = 1;
  cv::String window_original_image = std::to_string(window_id) + " :Original image";
  window_id++;

  cv::String window_artv_image = std::to_string(window_id) + " :AR TV";
  window_id++;

  if (test == 1){
      std::cout << "READING ARUCO TEST VIDEO" << std::endl;
      capdev = new cv::VideoCapture("./aruco_test_video.avi"); 
      if (!capdev->isOpened()){
        std::cout << "Error opening video stream or image file" << std::endl;
      }
  }

  while (true) {
    // std::cout << "####################### REACHED START OF WHILE LOOP #######################" << std::endl;
    // std::cout << "Frame before input from camera = " << std::endl << " " << frame << std::endl << std::endl;
    // std::cout << "Getting data from camera" << std::endl;
    
    
    if (test == 1){
      bool isSuccess = capdev->read(frame);

      // If frames are not there, close it
      if (isSuccess == false){
        std::cout << "Video file has ended. Running the video in loop" << std::endl;
        capdev->set(1,0);
        capdev->read(frame); //Src: https://stackoverflow.com/questions/17158602/playback-loop-option-in-opencv-videos
      }
    }
    else{
      *capdev >> frame; //frame.type() is 16 viz. 8UC3
    }
    
    cv::resize(frame, frame, cv::Size(res_width, res_height));    

    if( ar_source == 1){
      bool isSuccess = capdev2->read(frame_ar);

      // If frames are not there, close it
      if (isSuccess == false){
        std::cout << "Video file has ended. Running the video in loop" << std::endl;
        capdev2->set(1,0);
        capdev2->read(frame_ar); //Src: https://stackoverflow.com/questions/17158602/playback-loop-option-in-opencv-videos
      }
    }else if( ar_source == 2){
      // std::cout << "Getting data from camera" << std::endl;
      frame_ar = frame;
    }
    cv::resize(frame_ar, frame_ar, cv::Size(res_width, res_height));    

    // printf("Resizing image to %dx%d\n", res_width, res_height);
    
    int key_pressed = cv::waitKey(1000/fps); //Gets key input from user. Returns -1 if key is not pressed within the given time. Here, 1 ms.
    // std::cout << "Pressed key is: " << key_pressed << std::endl;

    // ASCII table reference: http://sticksandstones.kstrom.com/appen.html
    if(key_pressed == 'q'){ //Search for the function's output if no key is pressed within the given time           
        //Wait indefinitely until 'q' is pressed. 113 is q's ASCII value  
        std::cout << "q is pressed. Exiting the program" << std::endl;
        cv::destroyWindow("1: Original_Image"); //destroy the created window
        return 0;
    }

    // Load the dictionary that was used to generate the markers.
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250); // Try 50, 100, 1000 as well
    
    // Initialize the detector parameters using default values
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
                
    // Declare the vectors that would contain the detected marker corners and the rejected marker candidates
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    
    // The ids of the detected markers are stored in a vector
    std::vector<int> markerIds;
                
    // Detect the markers in the image. Please not that the markers can be detected in any order. It is not fixed that id 0 will be detected first.
    cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    
    // draw detected markers: frame, corner id
    // cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
    
    // pts_src contains the 4 corners of the original frame in clockwise order. TL, TR, BR, BL
    std::vector<cv::Point2f> pts_src{cv::Point2f(0,0), cv::Point2f(res_width-1,0), cv::Point2f(res_width-1,res_height-1), cv::Point2f(0,res_height-1) };
    
    std::vector<cv::Point2f> BoxCoordinates;
    if (markerIds.size() == 4){

      // BoxCoordinates contains the 4 corners where we want to show another video/ image (i.e. using corners of 4 aruco markers). They also need to be stored the same way as pts_src
      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 0) - markerIds.begin()][0]); // TL(0 th) corner of the marker with ID 0 (printed on TL corner of the page)
      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 1) - markerIds.begin()][1]);// TR(1 st) corner of the marker with ID 1 (printed on TR corner of the page)
      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 3) - markerIds.begin()][2]);// TL(2 nd) corner of the marker with ID 3 (printed on BR corner of the page)
      BoxCoordinates.push_back(markerCorners[find(markerIds.begin(), markerIds.end(), 2) - markerIds.begin()][3]);// TL(3 rd) corner of the marker with ID 2 (printed on BL corner of the page)

      std::cout << "Marker id: ";
      for (auto it:markerIds){
         std::cout << it << " "; 
      }
      std::cout << std::endl;

      std::cout << "Detected corners: ";
      for (auto it:BoxCoordinates){
        std::cout << it << " "; 
      }
      std::cout << std::endl;
      
      //Resize frame_ar wrt the dimensions of the bounding box
      // int req_width = sqrt((BoxCoordinates[0].x-BoxCoordinates[1].x)*(BoxCoordinates[0].x-BoxCoordinates[1].x) + (BoxCoordinates[0].y-BoxCoordinates[1].y)*(BoxCoordinates[0].y-BoxCoordinates[1].y));
      // int req_height = sqrt((BoxCoordinates[1].x-BoxCoordinates[2].x)*(BoxCoordinates[1].x-BoxCoordinates[2].x) + (BoxCoordinates[1].y-BoxCoordinates[2].y)*(BoxCoordinates[1].y-BoxCoordinates[2].y));

     
      // cv::imshow("Original frame_ar:", frame_ar);
      // std::cout << "Original frame_ar size: " << frame_ar.size() << std::endl;
      // std::cout << "Required width: " << req_width << " | Req height: " << req_height << std::endl;
      // cv::resize(frame_ar, frame_ar, cv::Size(req_width, req_height));
      // std::cout << "Resized frame_ar size: " << frame_ar.size() << std::endl;
      // cv::imshow("Resized frame_ar:", frame_ar);

      
      // std::cout << "pts_src 0th points is: " << pts_src[1] <<  std::endl;

      // METHOD 1: Compute homography from ar_source and destination points
      // Src: https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/
      
      cv::Mat h = cv::findHomography(pts_src, BoxCoordinates);

      // float x_offset = 0;
      // float y_offset = -50;
      
      // std::cout << "Homography matrix is: " << h << std::endl;

      // METHOD 2 | // Src: https://kediarahul.medium.com/augmented-reality-television-with-aruco-markers-opencv-python-c-c81823fbff54, 
      // cv::Mat h(2, 4, CV_32FC1);
      // h = cv::getPerspectiveTransform(pts_src, BoxCoordinates);

      // cv::Mat h_multiplier = (cv::Mat_<double>(3,3) << 1, 0, x_offset, 0, 1, y_offset, 0, 0, 1);
      // h = h_multiplier*h;




      // Warped image
      cv::Mat warpedImage;
                  
      // Warp ar_source image to destination based on homography
      cv::warpPerspective(frame_ar, warpedImage, h, frame.size(), cv::INTER_CUBIC);
      // cv::warpPerspective(frame_ar, warpedImage, h, cv::Size(300, res_width));//, cv::INTER_CUBIC);
      cv::imshow("warpPerspective output", warpedImage);       

      std::vector<cv::Point> BoxCoordinates_Converted;
      for (std::size_t i = 0; i < BoxCoordinates.size(); i++)
          BoxCoordinates_Converted.push_back(cv::Point(BoxCoordinates[i].x, BoxCoordinates[i].y));

      // Prepare a mask representing region to copy from the warped image into the original frame.
      cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
      cv::imshow("Mask original", mask);
      // Src: https://stackoverflow.com/questions/43629978/opencv-exception-fillconvexppoly
      cv::fillConvexPoly(mask, BoxCoordinates_Converted, cv::Scalar(255,255,255));
      // Mask now contains 255 (white) inside the bounding box rectangle and 0 (black) everywhere else

      // cv::circle(mask, )
      cv::imshow("Mask after filling", mask);

      cv::Mat mask_not;
      cv::bitwise_not(mask, mask_not);
      // mask_not now contains 0 (black) inside the bounding box rectangle and 255 (white) everywhere else
      cv::imshow("Mask not is", mask_not);
      std::cout << " ########### REACHED HERE ###########" << std::endl;
                  
      // Erode the mask to not copy the boundary effects from the warping
      cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3,3) );
      cv::erode(mask, mask, element);
      // cv::imshow("Mask after eroding", mask); //almost the same as mask before eroding
      // std::cout << "Mask is: " << mask << std::endl;
      // // Copy the masked warped image into the original frame in the mask region.
      // cv::Mat imOut = frame_ar.clone();
      cv::Mat frame_masked, overlapped_frame;
      cv::bitwise_and(frame, mask_not, frame_masked);
      cv::imshow("Frame masked", frame_masked); //almost the same as mask before eroding

      overlapped_frame = frame_masked.clone();
      cv::bitwise_or(frame_masked, warpedImage, overlapped_frame);

      // frame.copyTo(imOut, mask);
      cv::imshow(window_artv_image, overlapped_frame);
    }
    
    cv::imshow(window_original_image, frame);
  }
  return 0;
}
