/*Dev Vaibhav, Siddharth Maheshwari, Skanda Akkihebbal Prasanna
Spring 2023 CS 5330
Final Project: Autonomous Lane and Number plate detection using classic Computer Vision and Deep Learning 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h> // To use sleep functionality
#include "helper_functions.h"
#include <set>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <string>


// Converts char[] to std::string
std::string char_to_String(char* a)
{
  //Ref: https://www.geeksforgeeks.org/convert-character-array-to-string-in-c/
    std::string s(a);

    //Usage:
    // char a[] = { 'C', 'O', 'D', 'E' };
    // char b[] = "geeksforgeeks";
 
    // string s_a = convertToString(a);
    // string s_b = convertToString(b);
 
    // cout << s_a << endl;
    // cout << s_b << endl;
 
    // we cannot use this technique again
    // to store something in s
    // because we use constructors
    // which are only called
    // when the string is declared.
 
    // Remove commented portion
    // to see for yourself
 
    /*
    char demo[] = "gfg";
    s(demo); // compilation error
    */
 
    return s;
}

// Writes text (label) at the given location left, top over the input_image
void draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    // std::cout << "Baseline: " << baseLine << std::endl;
    top = std::max(top, label_size.height);
    // Top left corner.
    // Point tlc = Point(left, top);
    cv::Point tlc = cv::Point(left, top - baseLine);
    // Bottom right corner.
    // Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    cv::Point brc = cv::Point(left + label_size.width, top - label_size.height - 2*baseLine);
    // Draw black rectangle.

    // Compute the position of the text within the bounding box
    cv::Point textPosition(left, top - label_size.height);

    cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    // putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
    cv::putText(input_image, label, textPosition, FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

// Feed the image to the network and return the output
std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net)
{
    // Convert to blob.
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
 
    net.setInput(blob);
 
    // Forward propagate.
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
 
    return outputs;
}

// Process the output from the network and draw AABB on detected objects, perform NMS, show labels with confidence values, 
cv::Mat post_process(cv::Mat &input_image, int &nc, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name, std::vector<cv::Rect> &boxes_NMS, std::vector<std::string> &labels_NMS)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
    // const int dimensions = 85;
    const int dimensions = nc + 5;
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        // data += 85;
        data += nc + 5;
    }
    // Perform Non-Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        // cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Get the label for the class name and its confidence.
        std::string label = cv::format("%.2f", confidences[idx]);
        // std::string label = std::to_string(confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        boxes_NMS.push_back(box);
        labels_NMS.push_back(label);

        // Draw class labels.
        // draw_label(input_image, label, left, top);
    }
    return input_image;
}


// Perform Optical Character Recognition (OCR) on the provided image (src) and store the result in outText
int ocrTOtext(cv::Mat& src, std::string& outText){
    // Convert input image to grayscale
    cv::Mat grayscale_image;
    cv::cvtColor(src, grayscale_image, cv::COLOR_BGR2GRAY);

    // Threshold the grayscale image using Otsu's method
    cv::Mat thresholded_image;
    cv::threshold(grayscale_image, thresholded_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Find contours of each connected component in the thresholded image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Sort contours from left to right
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
        return cv::boundingRect(contour1).x < cv::boundingRect(contour2).x;
    });

    // Concatenate recognized characters to form the number plate
    std::string number_plate;
    for (const auto& contour : contours) {
        cv::Rect bounding_box = cv::boundingRect(contour);

        // Extract the character from the bounding box
        cv::Mat character_image = grayscale_image(bounding_box);
        cv::rectangle(character_image, bounding_box, cv::Scalar(0), 1);
        cv::imshow("char images",character_image);
        // Use Tesseract OCR to recognize the character
        tesseract::TessBaseAPI ocr;
        ocr.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
        ocr.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
        ocr.SetImage(character_image.data, character_image.cols, character_image.rows, 1, character_image.step);
        std::string recognized_text = ocr.GetUTF8Text();
                // Draw the contours around the character
        std::vector<std::vector<cv::Point>> contour_list = { contour };
        cv::drawContours(character_image, contour_list, -1, cv::Scalar(0, 255, 0), 2);
        // std::cout<< " recog char: "<< recognized_text<< std::endl;
        // Append the recognized character to the number plate
        outText += recognized_text;
    }

    // Display the input image with bounding boxes around each character
    cv::Mat input_image_with_boxes = src.clone();
    for (const auto& contour : contours) {
        cv::Rect bounding_box = cv::boundingRect(contour);
        // cv::rectangle(input_image_with_boxes, bounding_box, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Input Image with Bounding Boxes", input_image_with_boxes);

    // Display the final number plate
    std::cout << "Number Plate: " << outText << std::endl;
    number_plate = outText;
    // // Combine the edge and thresholded images
    // tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    // //This line initializes the Tesseract OCR engine with the English language and LSTM OCR engine mode.
    // ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    // //This line sets the page segmentation mode of the Tesseract OCR engine to automatic.
    // ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
    // //It uses the SetImage() function of the TessBaseAPI class and passes the image data, width, height, number of channels, and step size of the image.
    // ocr->SetImage(src.data, src.cols, src.rows, 3, src.step);
    // ocr->SetSourceResolution(70);
    // // ocr->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    // //runs the OCR on the image using the GetUTF8Text() function of the TessBaseAPI class and assigns the recognized text to the outText string variable.
    // outText = std::string(ocr->GetUTF8Text());
    // // std::cout <<"detected: "<< outText;
    // // cv:: imshow("src", src);
    // ocr->End();
    return(0);
}

// Improve contrast and illumination
void illuminationCorrection(cv::Mat& image) {
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

    // Split the LAB image into its 3 channels
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);

    // Apply CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(lab_planes[0], lab_planes[0]);

    // Merge the processed LAB channels
    cv::merge(lab_planes, lab_image);

    // Convert the LAB image back to BGR color space
    cv::Mat corrected_image;
    cv::cvtColor(lab_image, corrected_image, cv::COLOR_Lab2BGR);

    // Replace the original image with the corrected one
    image = corrected_image;
}

// Calculate 3D histogram
int calc_histogram3D(cv::Mat &src){
    // Src: https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html#code
    /// Separate the image in 3 places ( B, G and R )
    std::vector<cv::Mat> bgr_planes;
    cv::split( src, bgr_planes );



    /// Establish the number of bins
    int histSize_h = 180;
    int histSize_sv = 255;

    /// Set the ranges ( for B,G,R) )
    float range_h[] = { 0, 180 } ;
    float range_sv[] = { 0, 256 } ;
    const float* histRange_h = { range_h };
    const float* histRange_sv = { range_sv };

    bool uniform = true; bool accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize_h, &histRange_h, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize_sv, &histRange_sv, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize_sv, &histRange_sv, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize_sv );

    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    // normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    // normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    // normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    /// Draw for each channel
    // for( int i = 1; i < histSize_sv; i++ )
    // {
    //     line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
    //                     cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
    //                     cv::Scalar( 255, 0, 0), 2, 8, 0  );
    //     line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
    //                     cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
    //                     cv::Scalar( 0, 255, 0), 2, 8, 0  );
    //     line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
    //                     cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
    //                     cv::Scalar( 0, 0, 255), 2, 8, 0  );
    // }

    // Find the max value from b_hist (corresponds to H in HSV)
    std::cout << "B_hist shape: " << b_hist.size << std::endl;

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(b_hist, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << b_hist << std::endl;

    std::cout << "H Maximum value: " << maxVal << std::endl;
    std::cout << "S corresponding to H Maximum value: " << g_hist.at<float>(maxLoc) << std::endl;
    std::cout << "V corresponding to V max value: " << r_hist.at<float>(maxLoc) << std::endl;
    std::cout << "Location of maximum value: " << maxLoc << std::endl;

    std::cout << "H Maximum value: " << maxVal << std::endl;
    std::cout << "S corresponding to H Maximum value: " << g_hist.at<float>(60) << std::endl;
    std::cout << "V corresponding to V max value: " << r_hist.at<float>(maxLoc) << std::endl;


    /// Display
    cv::imshow("calcHist Demo", histImage );

    return 0;
}

// SKANDA HELPER FUNCTIONS

cv::Mat applyGaussianBlur(cv::Mat &input) {
    cv::Mat output;
    cv::GaussianBlur(input, output, cv::Size(3, 3), 0, 0);
    return output;
}

cv::Mat convertToGrayscale(cv::Mat &input) {
    cv::Mat output;
    cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
    return output;
}


cv::Mat applyCanny(cv::Mat &input, int lowThreshold, int ratio, int kernel_size) {
    cv::Mat output;
    cv::Canny(input, output, lowThreshold, lowThreshold * ratio, kernel_size);
    return output;
}


void warpPerspective(cv::Mat& frame) {
    int height = frame.rows;
    int width = frame.cols;
    int offset = 50;

    cv::Point2f srcPoints[4], dstPoints[4];
    srcPoints[0] = cv::Point(width * 0.46, height * 0.72);
    srcPoints[1] = cv::Point(width * 0.58, height * 0.72);
    srcPoints[2] = cv::Point(width * 0.30, height);
    srcPoints[3] = cv::Point(width * 0.82, height);

    dstPoints[0] = cv::Point(offset, 0);
    dstPoints[1] = cv::Point(width - 2 * offset, 0);
    dstPoints[2] = cv::Point(offset, height);
    dstPoints[3] = cv::Point(width - 2 * offset, height);

    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);

    cv::warpPerspective(frame, frame, M, frame.size());
}

std::pair<int, int> histogram(cv::Mat &frame) {
    // Build histogram
    cv::Mat histogram = frame.clone();
    // std::cout<<hi
    int histSize = 256;  // Number of bins
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::calcHist(&frame, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange);

    // Find mid point on histogram
    int midpoint = histogram.cols / 2;
    // std::cout<<"MidPoint: "<<midpoint<<std::endl;

    // Compute the left max pixels
    cv::Mat left_half = histogram.colRange(0, midpoint);
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(left_half, &min_val, &max_val, &min_loc, &max_loc);
    int left_x_base = max_loc.x;

    // Compute the right max pixels
    cv::Mat right_half = histogram.colRange(midpoint, histogram.cols);
    cv::minMaxLoc(right_half, &min_val, &max_val, &min_loc, &max_loc);
    int right_x_base = max_loc.x + midpoint;

    return { left_x_base, right_x_base };
}

std::vector<cv::Vec4i> detect_lines(cv::Mat frame) {
    // Find lines on the smaller frame using Hough Lines Polar
    std::vector<cv::Vec4i> line_segments;
    cv::HoughLinesP(frame, line_segments, 1, CV_PI/180, 20, 40, 150);
    return line_segments;   // Return line segment on road
}


std::vector<cv::Vec4i> map_coordinates(cv::Mat &frame, std::pair<double, double> parameters) {
    int height = frame.rows;
    int width = frame.cols;

    double slope = parameters.first;
    double intercept = parameters.second;

    if (slope == 0) {
    // std::cout<<"slope"<<slope<<std::endl;
        slope = 0.1;
    }

    int y1 = height;
    int y2 = static_cast<int>(height * 0.72);
    int x1 = static_cast<int>((y1 - intercept) / slope);
    int x2 = static_cast<int>((y2 - intercept) / slope);

    std::vector<cv::Vec4i> coords = {{x1, y1, x2, y2}};
    return coords;
}

std::vector<cv::Vec4i> convert_lines(const std::vector<std::vector<int>>& lines) {
    std::vector<cv::Vec4i> converted_lines;
    for (const auto& line : lines) {
        cv::Vec4i converted_line(line[0], line[1], line[2], line[3]);
        converted_lines.push_back(converted_line);
    }
    return converted_lines;
}
std::vector<cv::Vec4i> optimize_lines(cv::Mat &frame, std::vector<cv::Vec4i> lines) {
    int height = frame.rows;
    int width = frame.cols;

    std::vector<cv::Vec4i> lane_lines;

    std::vector<std::pair<double, double>> left_fit;
    std::vector<std::pair<double, double>> right_fit;

    for (auto line : lines) {
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];
        // std::cout<<"x1: "<<x1<<" y1: "<<y1<<" x2: "<<x2<<" y2: "<<y2<<std::endl;
        // Calculate the slope and intercept of the line
        double slope = (y2 - y1) /double(x2 - x1);

        // std::cout<<"slope"<<slope<<std::endl;
        double intercept = y1 - slope * x1;
        // Store the slope and intercept in a pair
        std::pair<double, double> parameters = std::make_pair(slope, intercept);
        if (slope < 0) {
            left_fit.push_back(parameters);
        } else {
            right_fit.push_back(parameters);
        }
    }

    if (left_fit.size() > 0) {
        std::pair<double, double> left_fit_average(0.0, 0.0);
        for (auto fit : left_fit) {
            left_fit_average.first += fit.first;
            left_fit_average.second += fit.second;
        }
        left_fit_average.first /= left_fit.size();
        left_fit_average.second /= left_fit.size();
        for (auto point : map_coordinates(frame, left_fit_average)) { lane_lines.push_back(point); }

    }

    if (right_fit.size() > 0) {
        std::pair<double, double> right_fit_average(0.0, 0.0);
        for (auto fit : right_fit) {
            right_fit_average.first += fit.first;
            right_fit_average.second += fit.second;
        }
        right_fit_average.first /= right_fit.size();
        right_fit_average.second /= right_fit.size();
        for (auto point : map_coordinates(frame, right_fit_average)) { lane_lines.push_back(point); }
    }

    return lane_lines;
}

cv::Mat display_lines(cv::Mat& frame, std::vector<cv::Vec4i>& lines) { 
    // Create a mask with zeros using the same dimension as frame.
    cv::Mat mask(frame.size(), CV_8UC3, cv::Scalar(0)); 
    // Check if there is a line. 
    if (!lines.empty()) {
        for (const auto& line : lines) 
        { // Draw the line on the created mask. 
        cv::line(mask, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 5);
        } 
        } 
        // Merge mask with original frame. 
        cv::Mat result;
        cv::addWeighted(frame, 0.8, mask, 1, 1, result);
        return result;
}
std::pair<int, int> get_floating_center(cv::Mat &frame, std::vector<cv::Vec4i> lane_lines) {
    int height = frame.rows;
    int width = frame.cols;
    int left_x1, left_x2, right_x1, right_x2;

    if (lane_lines.size() == 2) {
        left_x1 = lane_lines[0][0];
        left_x2 = lane_lines[0][2];
        right_x1 = lane_lines[1][0];
        right_x2 = lane_lines[1][2];

        int low_mid = (right_x1 + left_x1) / 2;
        int up_mid = (right_x2 + left_x2) / 2;

        return {up_mid, low_mid};
    }
    else {
        int up_mid = static_cast<int>(width * 1.9);
        int low_mid = static_cast<int>(width * 1.9);
        // std::cout<<"up mid: "<<up_mid<<std::endl;
        // std::cout<<"low mid: "<<low_mid<<std::endl;
        
        return {up_mid, low_mid};
    }
}

cv::Mat add_text(cv::Mat frame, int image_center, int left_x_base, int right_x_base) {
    // std::cout<<" lef base : "<<left_x_base<<std::endl;
    // std::cout<<" rigt base : "<<right_x_base<<std::endl;

    double lane_center = left_x_base + (right_x_base - left_x_base) / 2;
    // std::cout<<" lane center : "<<lane_center<<std::endl;
    double deviation = image_center - lane_center;
    // std::cout<<" deviation: "<<deviation<<std::endl;
    std::string text;
    if (deviation > 160) {
        text = "Smooth Left";
    } else if (deviation < 40 || (deviation > 150 && deviation <= 160)) {
        text = "Smooth Right";
    } else if (deviation >= 40 && deviation <= 150) {
        text = "Straight";
    }
    cv::putText(frame, "DIRECTION: " + text, cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    return frame;
}

// Detects lane using classical CV
int lane_detection(cv::Mat &src, cv::Mat &dst, cv::Mat &mask){
    
    cv::Mat grayVideo;
    cv::Mat cannyedgeVideo;

    int width = src.cols;
    int height = src.rows;
    // std::cout << "Video size: " << width << " x " << height << std::endl;
    
    // denoising
    cv::Mat blur_video = applyGaussianBlur(src);
    // cv::imshow("Gaussian blur video", blur_video);
    // grayscale of denoised frame
    grayVideo = convertToGrayscale(blur_video);
        // cv::imshow("gray_scale_after_blur", grayVideo);
// canny edge detection
    cannyedgeVideo = applyCanny(grayVideo, 50, 3, 3);
    // cv::imshow("cannyVideo", cannyedgeVideo);        
// fill poly --> masking
        // createMask(mask, cannyedgeVideo, width, height);
    // cv::Mat mask = createMask(cannyedgeVideo, width, height);
    // Read the sixth frame from the video
    cv::Mat frame, sixth_frame;
    // for (int i = 0; i < 6; i++) {
    //     cap.read(src);
    // }
    sixth_frame =src.clone();

    
    
    cv::imshow("Mask", mask);


    // cv::imshow("mask", mask);
    // std::cout << "Width of mask: " << mask.cols << ", Height of mask: " << mask.rows << std::endl;
    // cv::Mat cropped_edges = mask.clone();

    cv::Mat cropped_edges = cv::Mat::zeros(mask.size(), mask.type()); //It should be single channel
    
    std::cout << "Mask channels: " << mask.channels() << std::endl; 
    std::cout << "cropped_edges channels: " << cropped_edges.channels() << std::endl; 
    std::cout << "grayVideo channels: " << grayVideo.channels() << std::endl; 



    cv::bitwise_and(grayVideo, mask, cropped_edges);
    std::cout << "Created bitwise_and" << std::endl;


    cv::imshow("crop edge Video", cropped_edges);
    // cv::waitKey(0);

    cv::Mat canny_masked = cv::Mat::zeros(mask.size(), mask.type());
    cv::bitwise_and(cannyedgeVideo, mask, canny_masked);
    std::cout << "Created cannyedgeVideo" << std::endl;
    // cv::imshow("canny masked Video", canny_masked);

    warpPerspective(cannyedgeVideo);
    std::cout << "Created warpPerspective" << std::endl;
    // cv::imshow("after warp cannyVideo", cannyedgeVideo);        

    cv::Mat warped_image = src.clone();  // Make a copy of the input image

    warpPerspective(warped_image);  // Warp the image
    // cv::imshow("warped_image", cropped_edges);  // Show the warped image

    // cv::Mat warped_image_gray;
    // cv::cvtColor(cannyedgeVideo, warped_image_gray, cv::COLOR_BGR2GRAY);

    std::pair<int, int> bases = histogram(cannyedgeVideo);
    int left_x_base = bases.first;
    int right_x_base = bases.second;

        std::vector<cv::Vec4i> line_segments = detect_lines(canny_masked); // detect the lines
    // std::cout<<"line segment started"<<std::endl;
    for (auto line_segment : line_segments) { 
        int x1 = line_segment[0];
        int y1 = line_segment[1];
        int x2 = line_segment[2];
        int y2 = line_segment[3];
        // std::cout<<"line_segment"<<std::endl;
    // cv::line(src, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
    }
    // std::cout<<"line segment completed"<<std::endl;
    // cv::imshow("Hough",src);

            // cv::imshow("detected_lines", warped_image_gray); // display the image with the detected
        std::vector<cv::Vec4i> optimized_lines = optimize_lines(cannyedgeVideo, line_segments);
        // std::vector<cv::Vec4i> optimized_lines = optimize_lines(src, line_segments);
        cv::Mat outputFrame = display_lines(src, optimized_lines);
        std::pair<int, int> shifting_points = get_floating_center(outputFrame, optimized_lines);
        int imageCenter = outputFrame.cols / 2;
        dst = add_text(outputFrame, shifting_points.first, left_x_base, right_x_base);
        // cv::imshow("Frame with Text", frameWithText);
    return 0;
}

// Detect objects in the image (src) using a network (net) having classes (class_list) where nc is the number of classes, stores the result in dst(shows AABB and label)
int detect_objects(cv::Mat &src, int &nc, std::vector<std::string> &class_list, cv::dnn::Net &net, cv::Mat &dst){
    
    // Process the image.
    std::vector<cv::Mat> detections;     
    detections = pre_process(src, net);
    cv::Mat src_cloned = src.clone();
    std::vector<cv::Rect> boxes_NMS;
    std::vector<std::string> labels_NMS;
    
    dst = post_process(src_cloned, nc, detections, class_list, boxes_NMS, labels_NMS);
    for (int i = 0; i < boxes_NMS.size(); i++)
    {
        // Draw bounding box.
        cv::rectangle(dst, cv::Point(boxes_NMS[i].x, boxes_NMS[i].y), cv::Point(boxes_NMS[i].x + boxes_NMS[i].width, boxes_NMS[i].y + boxes_NMS[i].height), BLUE, 3*THICKNESS);
        // Draw class labels.
        draw_label(dst, labels_NMS[i], boxes_NMS[i].x, boxes_NMS[i].y);
        // std::cout << "x: " <<  boxes_NMS[i].x << std::endl;
        // std::cout << "y: " <<  boxes_NMS[i].y << std::endl;
        // std::cout << "width: " <<  boxes_NMS[i].width << std::endl;
        // std::cout << "height: " <<  boxes_NMS[i].height << std::endl;
        // std::cout << "src size: " << src.size << std::endl;
        // std::cout << "y range " << boxes_NMS[i].y + boxes_NMS[i].height << std::endl;
        // std::cout << "x range " << boxes_NMS[i].x + boxes_NMS[i].width << std::endl;



        // cv::Mat plate_img = src(cv::Range(100 , 150), cv::Range(200,300));
        
        // Feed the bounding box information to tesseract to do OCR
        int buffer = 0;
        // std::cout << "Starting ROI calculation" << std::endl;
        // std::cout << "x: " << boxes_NMS[i].x << " | y: " << boxes_NMS[i].y << " | width: " << boxes_NMS[i].width << " | height: " << boxes_NMS[i].height << std::endl;

        int x_max = boxes_NMS[i].x + boxes_NMS[i].width;
        int y_max = boxes_NMS[i].y + boxes_NMS[i].height;

        int x_min = boxes_NMS[i].x;
        int y_min = boxes_NMS[i].y;

        if (x_min < 0){
            x_min = 0;
        }

        if (y_min < 0){
            y_min = 0;
        }

        if (boxes_NMS[i].x + boxes_NMS[i].width >= src.cols){
            x_max = src.cols - 1;
        }
        if (boxes_NMS[i].y + boxes_NMS[i].height >= src.rows){
            y_max = src.rows - 1;
        }

        cv::Rect roi(x_min, y_min, x_max - boxes_NMS[i].x , y_max - boxes_NMS[i].y);
        cv::Mat plate_img = src(roi);
        // std::cout << "Calculated ROI" << std::endl;
        // cv::imshow("plate img", plate_img);
        
        if (nc == 1){
            // std::cout << "Performing OCR" << std::endl;
            std::string ocr_text;

            // cv::Mat test_img = cv::imread("bmw.jpeg");
            // cv::Size img_size = test_img.size();

            // Print image width and height
            // std::cout << "Image width: " << img_size.width << std::endl;
            // std::cout << "Image height: " << img_size.height << std::endl;



            // Pre-processing on number plate
            
            illuminationCorrection(plate_img);
            // cv::imshow("Ill correction",plate_img);
            
            // cv::Mat plate_img_hsv;
            // cv::cvtColor(plate_img, plate_img_hsv, cv::COLOR_BGR2HSV);
            // cv::imshow("HSV", plate_img_hsv);

            
            // cv::cvtColor(plate_img, plate_img, cv::COLOR_BGR2GRAY);
            // cv::imshow("gray",plate_img);


            // int histSize = 255;
            // float range[] = { 0, 256 } ;
            // const float* histRange = { range };
            // cv::Mat grey_hist;
            // cv::calcHist( &plate_img, 1, 0, cv::Mat(), grey_hist, 1, &histSize, &histRange, true, false );

            // // Draw the histograms for B, G and R
            // int hist_w = 512; int hist_h = 400;
            // int bin_w = cvRound( (double) hist_w/histSize );

            // cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
            // // Draw for each channel
            // for( int i = 1; i < histSize; i++ )
            // {
            //     cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(grey_hist.at<float>(i-1)) ) ,
            //                     cv::Point( bin_w*(i), hist_h - cvRound(grey_hist.at<float>(i)) ),
            //                     cv::Scalar( 255, 0, 0), 2, 8, 0  );
            // }

            // cv::imshow("calcHist Demo", histImage );


            // double minVal, maxVal;
            // cv::Point minLoc, maxLoc;
            // cv::minMaxLoc(grey_hist, &minVal, &maxVal, &minLoc, &maxLoc);

            // std::cout << grey_hist << std::endl;

            // std::cout << "H Maximum value: " << maxVal << std::endl;
            // std::cout << "S corresponding to H Maximum value: " << g_hist.at<float>(maxLoc) << std::endl;
            // std::cout << "V corresponding to V max value: " << r_hist.at<float>(maxLoc) << std::endl;
            // std::cout << "Location of maximum value: " << maxLoc << std::endl;

            // Calculate histogram
            // calc_histogram3D(plate_img_hsv);
            
            // cv::GaussianBlur(plate_img, plate_img, cv::Size(3, 3), 0);
            // cv::imshow("Gaussian Blur",plate_img);

            // cv::cvtColor(plate_img, gray, cv::COLOR_BGR2GRAY);
            // cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX);

            // cv::threshold(plate_img, plate_img, 0, 255, cv::THRESH_BINARY+cv::THRESH_OTSU);
            // cv::adaptiveThreshold(plate_img, plate_img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11,2);
            
            // cv::adaptiveThreshold(plate_img, plate_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11,2);

            // cv::imshow("Thresholded adaptive",plate_img);

            // Apply morphological operators for noise reduction
            // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            // cv::dilate(plate_img, plate_img, kernel);
            // cv::erode(plate_img, plate_img, kernel);
            // cv::dilate(plate_img, plate_img, kernel);
            // cv::dilate(plate_img, plate_img, kernel);

            // cv::dilate(edge, edge, kernel);
            // Apply thresholding to the dilated image
            // cv::imshow("erode ",plate_img);
            
            // cv::bitwise_not(thresh, thresh);
            // Apply morphological operators
            // kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
            // cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);
            // cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

            ocrTOtext(plate_img,ocr_text);
            // Clean-up the detected text. Remove trailing \n
            size_t pos = ocr_text.find_last_of("\n");
            if (pos != std::string::npos) {
                ocr_text.erase(pos);
            }
            
            // Draw class labels.
            draw_label(dst, ocr_text, boxes_NMS[i].x, boxes_NMS[i].y + 2*boxes_NMS[i].height + 5);

            // for (char c : ocr_text) {
            //     std::cout << (int)c << " ";
            // }
            // cv::imshow("Original Image", img);
            // cv::imshow("Processed Image", thresh);
            std::cout << "Detected text is:" << ocr_text << "text_ended" << std::endl;
            // cv::waitKey(0);
        }
        
        
    }

    
    
    // Put efficiency information.
    // The function getPerfProfile returns the overall time for     inference(t) and the timings for each of the layers(in layersTimes).
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time : %.2f ms", t);
    // std::string label = "Inference time";
    cv::putText(dst, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    return 0;
}

cv::Mat createMask(cv::Mat &myImage, std::vector<cv::Point>& vertices) {
    cv::Mat mask = cv::Mat::zeros(myImage.size(), CV_8UC1);
    const cv::Point* ppt[1] = { &vertices[0] };
    int npt[] = { static_cast<int>(vertices.size()) };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255));
    return mask;
}

// Callback function for mouse events
void selectROI(int event, int x, int y, int flags, void* userdata) {
    
    // std::cout << "Entered select ROI" << std::endl;
    struct myMouseCallbackData{
        cv::Mat frame;
        std::vector<cv::Point> vertices;
    };
    // cv::Mat frame = *((cv::Mat*)userdata);
    
    myMouseCallbackData* data = (myMouseCallbackData*)(userdata);

    // std::cout << "Reached before while" << std::endl;
    
    // while (data->vertices.size() < 4){
        if (event == cv::EVENT_LBUTTONDOWN) {
            data->vertices.push_back(cv::Point(x, y));
            cv::circle(data->frame, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);

            cv::imshow("Select ROI", data->frame);
        } else if (event == cv::EVENT_RBUTTONDOWN) {
            // if (vertices.size() > 0) {
            //     savePointsToFile(vertices, "home/skanda/Documents/prcv_final/input/roi_points.txt");
            // }
            data->vertices.clear();
            // cv::imshow("Select ROI", data->frame);
        }
    // }

    // std::cout << "Reached after while" << std::endl;

    // if (data->vertices.size() == 4) {
    //         // std::vector<cv::Point> pts{ data->vertices };
    //         data->mask = createMask(data->frame, data->vertices);
    //         // cout << "Updated mask:\n" << mask << endl; // print the updated mask
    // }

    // std::cout << "Mask calculated" << std::endl;
    
}
