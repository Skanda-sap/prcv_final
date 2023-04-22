/*Dev Vaibhav, Siddharth Maheshwari, Skanda Akkihebbal Prasanna
Spring 2023 CS 5330
Final Project: Autonomous Lane and Number plate detection using classic Computer Vision and Deep Learning 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <unistd.h> // To use sleep functionality
#include "helper_functions.h"
#include "filter.h"
#include <set>
#include <dirent.h>

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

void draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    std::cout << "Baseline: " << baseLine << std::endl;
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

cv::Mat post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name, std::vector<cv::Rect> &boxes_NMS, std::vector<std::string> &labels_NMS)
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
        // std::string label = format("%.2f", confidences[idx]);
        std::string label = std::to_string(confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        boxes_NMS.push_back(box);
        labels_NMS.push_back(label);

        // Draw class labels.
        // draw_label(input_image, label, left, top);
    }
    return input_image;
}



int ocrTOtext(cv::Mat& im, std::string& outText){
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    //This line initializes the Tesseract OCR engine with the English language and LSTM OCR engine mode.
    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    //This line sets the page segmentation mode of the Tesseract OCR engine to automatic.
    ocr->SetPageSegMode(tesseract::PSM_AUTO);
    //It uses the SetImage() function of the TessBaseAPI class and passes the image data, width, height, number of channels, and step size of the image.
    ocr->SetImage(im.data, im.cols, im.rows, 3, im.step);
    // ocr->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    //runs the OCR on the image using the GetUTF8Text() function of the TessBaseAPI class and assigns the recognized text to the outText string variable.
    outText = std::string(ocr->GetUTF8Text());
    std::cout <<"detected: "<< outText;
    ocr->End();
    return(0);

}


