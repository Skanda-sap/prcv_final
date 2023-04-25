/*Dev Vaibhav, Siddharth Maheshwari, Skanda Akkihebbal Prasanna
Spring 2023 CS 5330
Final Project: Autonomous Lane and Number plate detection using classic Computer Vision and Deep Learning 
*/
#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<math.h>



// **********************************************************************************
// Helper Functions
// **********************************************************************************

// Gaussian Blur Function
cv::Mat applyGaussianBlur(cv::Mat &input);
// Grayscale Conversion Function
cv::Mat convertToGrayscale(cv::Mat &input);
// CannyEdge detection
cv::Mat applyCanny(cv::Mat &input, int lowThreshold, int ratio, int kernel_size);
// Masking function
cv::Mat createMask(const cv::Mat &myImage, float width, float height);
// Warperspectivef function
void warpPerspective(cv::Mat& frame);
// histogram 
std::pair<int, int> histogram(cv::Mat &frame);
// detect lines
std::vector<cv::Vec4i> detect_lines(cv::Mat frame);
// map coorindate function
std::vector<cv::Vec4i> map_coordinates(cv::Mat &frame, std::pair<double, double> parameters);
// convert line function
std::vector<cv::Vec4i> convert_lines(const std::vector<std::vector<int>>& lines);
// optimize lines function
std::vector<cv::Vec4i> optimize_lines(cv::Mat &frame, std::vector<cv::Vec4i> lines);
// display lines function
cv::Mat display_lines(cv::Mat& frame, std::vector<cv::Vec4i>& lines);
// floating center function
std::pair<int, int> get_floating_center(cv::Mat &frame, std::vector<cv::Vec4i> lane_lines);
// add_text function
cv::Mat add_text(cv::Mat frame, int image_center, int left_x_base, int right_x_base);
#endif
