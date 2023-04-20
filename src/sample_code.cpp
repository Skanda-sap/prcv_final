// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/videoio.hpp"
// #include <iostream>

// using namespace cv;

// /** Global Variables */
// const int max_value_H = 360/2;
// const int max_value = 255;
// const String window_capture_name = "Video Capture";
// const String window_detection_name = "Object Detection";
// int low_H = 0, low_S = 0, low_V = 0;
// int high_H = max_value_H, high_S = max_value, high_V = max_value;

// //! [low]
// static void on_low_H_thresh_trackbar(int, void *)
// {
//     low_H = min(high_H-1, low_H);
//     setTrackbarPos("Low H", window_detection_name, low_H);
// }
// //! [low]

// //! [high]
// static void on_high_H_thresh_trackbar(int, void *)
// {
//     high_H = max(high_H, low_H+1);
//     setTrackbarPos("High H", window_detection_name, high_H);
// }

// //! [high]
// static void on_low_S_thresh_trackbar(int, void *)
// {
//     low_S = min(high_S-1, low_S);
//     setTrackbarPos("Low S", window_detection_name, low_S);
// }

// static void on_high_S_thresh_trackbar(int, void *)
// {
//     high_S = max(high_S, low_S+1);
//     setTrackbarPos("High S", window_detection_name, high_S);
// }

// static void on_low_V_thresh_trackbar(int, void *)
// {
//     low_V = min(high_V-1, low_V);
//     setTrackbarPos("Low V", window_detection_name, low_V);
// }

// static void on_high_V_thresh_trackbar(int, void *)
// {
//     high_V = max(high_V, low_V+1);
//     setTrackbarPos("High V", window_detection_name, high_V);
// }

// int main(int argc, char* argv[])
// {
//     //! [cap]
//     VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
//     //! [cap]

//     //! [window]
//     namedWindow(window_capture_name);
//     namedWindow(window_detection_name);
//     //! [window]

//     //! [trackbar]
//     // Trackbars to set thresholds for HSV values
//     createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
//     createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
//     createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
//     createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
//     createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
//     createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
//     //! [trackbar]

//     Mat frame, frame_HSV, frame_threshold;
//     while (true) {
//         //! [while]
//         cap >> frame;
//         if(frame.empty())
//         {
//             break;
//         }

//         // Convert from BGR to HSV colorspace
//         cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
//         // Detect the object based on HSV Range Values
//         inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
//         //! [while]

//         //! [show]
//         // Show the frames
//         imshow(window_capture_name, frame);
//         imshow(window_detection_name, frame_threshold);
//         //! [show]

//         char key = (char) waitKey(30);
//         if (key == 'q' || key == 27)
//         {
//             break;
//         }
//     }
//     return 0;
// }

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using std::cout;
const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;
Mat src1;
Mat src2;
Mat dst;
static void on_trackbar( int, void* )
{
   alpha = (double) alpha_slider/alpha_slider_max ;
   beta = ( 1.0 - alpha );
   addWeighted( src1, alpha, src2, beta, 0.0, dst);
   imshow( "Linear Blend", dst );
}
int main( void )
{
   src1 = imread( cv::samples::findFile("LinuxLogo.jpg") );
   src2 = imread( cv::samples::findFile("WindowsLogo.jpg") );
   if( src1.empty() ) { cout << "Error loading src1 \n"; return -1; }
   if( src2.empty() ) { cout << "Error loading src2 \n"; return -1; }
   alpha_slider = 0;
   namedWindow("Linear Blend", WINDOW_AUTOSIZE); // Create Window
   char TrackbarName[50];
   sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );
   createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );
   on_trackbar( alpha_slider, 0 );
   waitKey(0);
   return 0;
}