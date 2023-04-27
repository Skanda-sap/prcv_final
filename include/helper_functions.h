/*Dev Vaibhav, Siddharth Maheshwari, Skanda Akkihebbal Prasanna
Spring 2023 CS 5330
Final Project: Autonomous Lane and Number plate detection using classic Computer Vision and Deep Learning 
*/

#ifndef helper_functions
#define helper_functions


extern int camera_ID;

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


// Char to string conversion
std::string char_to_String(char* a);

// Draws label on the image
void draw_label(cv::Mat& input_image, std::string label, int left, int top);

// Preprocesses the image to feed to network
std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net);

// Postprocess the image, filter candidates for bounding box, perform NMS
cv::Mat post_process(cv::Mat &input_image, int &nc, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name, std::vector<cv::Rect> &boxes_NMS, std::vector<std::string> &labels_NMS);

// Do OCR on image
int ocrTOtext(cv::Mat& im, std::string& outText);

// Illumination correction
void illuminationCorrection(cv::Mat& image);

// Calculates 3D histogram
int calc_histogram3D(cv::Mat &src);

// Detects Lane using classical CV
int lane_detection(cv::Mat &src, cv::Mat &dst, cv::Mat &mask);

// Detect objects in image using neural network
int detect_objects(cv::Mat &src, int &nc, std::vector<std::string> &class_list, cv::dnn::Net &net, cv::Mat &dst);


// SKANDA HELPER FUNCTIONS


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

// Create mask function
cv::Mat createMask(cv::Mat &myImage, std::vector<cv::Point>& vertices);

// Region of intererst extraction function
void selectROI(int event, int x, int y, int flags, void* userdata);

#endif
