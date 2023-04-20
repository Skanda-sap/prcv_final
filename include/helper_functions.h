/*Dev Vaibhav
Spring 2023 CS 5330
Project 4: Calibration and Augmented Reality
*/

#ifndef helper_functions
#define helper_functions



extern cv::Size patternsize; //interior number of corners
// Resolution required for the window
extern int res_width; //columns
extern int res_height; //rows

extern const float INPUT_WIDTH;
extern const float INPUT_HEIGHT;
extern const float SCORE_THRESHOLD; // To filter low probability class scores.
extern const float NMS_THRESHOLD; // To remove overlapping bounding boxes.
extern const float CONFIDENCE_THRESHOLD; // Filters low probability detections.

// Text parameters.
extern const float FONT_SCALE;
extern const int FONT_FACE; // Its enumeration is zero | Src: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
extern const int THICKNESS;
 
// Colors.
extern cv::Scalar BLACK;
extern cv::Scalar BLUE;
extern cv::Scalar YELLOW;
extern cv::Scalar RED;

std::string char_to_String(char* a);

void draw_label(cv::Mat& input_image, std::string label, int left, int top);

std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net);

cv::Mat post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name);

#endif