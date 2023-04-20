/*Dev Vaibhav
Spring 2023 CS 5330
Project 4: Calibration and Augmented Reality| Using functions from project 1
*/

// function computing horizontal gradient
int greyscale( cv::Mat &src, cv::Mat &dst );

// function to compute Gausian blur
int blur5x5( cv::Mat &src, cv::Mat &dst );

// Sobel X and Y as separable filters. X: positive right; : positive up
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

// gradient magnitude image using Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

// blurs and quantizes a color image
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

// live video cartoonization function using the gradient magnitude and blur/quantize filters
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

// Function to calcuate the negative of a color image
int negative( cv::Mat &src, cv::Mat&dst);

// Function to sharpen color image
int sharpen( cv::Mat &src, cv::Mat&dst);

// Function to make color image's sketch
int sketch( cv::Mat &src, cv::Mat&dst);

// Adding caption to image
int caption(cv::Mat &src, cv::Mat&dst, std::string text);

// Rotating the image
int rotate(cv::Mat &src, cv::Mat&dst);

// L5E5 filter
int L5E5(cv::Mat &src, cv::Mat&dst);

// BGR to HSV converter
int bgr_to_hsv(cv::Mat &src, cv::Mat&dst);

// Implements Grassfire transform.
// Connectedness : 4 or 8
// Operation : 0 : Erosion | 1 : Dilation
// max_num_operations: Max. number of erosions or dilations to do in the third pass.
int grassfire_tf(cv::Mat &src, cv::Mat&dst, int connectedness, int operation, int max_num_operations);

// Draws the HSV histogram
// Input: 3 channels of the image (BGR or HSV)
void drawHistogram(cv::Mat& b_hist,cv::Mat& g_hist,cv::Mat& r_hist);

