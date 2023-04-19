#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include "/home/sid/Documents/prcv projects/tocr/include/helper.h"

int main(){ 

//load images
std::string outText, imPath = "/home/sid/Documents/prcv projects/tocr/img/ma.png";
cv:: Mat im = cv::imread(imPath, cv::IMREAD_COLOR);
// //This line declares a pointer to the TessBaseAPI class of Tesseract OCR and initializes it using the new operator to create a new instance of the class.
// tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
// //This line initializes the Tesseract OCR engine with the English language and LSTM OCR engine mode.
// ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
// //This line sets the page segmentation mode of the Tesseract OCR engine to automatic.
// ocr->SetPageSegMode(tesseract::PSM_AUTO);
// //It uses the SetImage() function of the TessBaseAPI class and passes the image data, width, height, number of channels, and step size of the image.
// ocr->SetImage(im.data, im.cols, im.rows, 3, im.step);
// // ocr->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
// //runs the OCR on the image using the GetUTF8Text() function of the TessBaseAPI class and assigns the recognized text to the outText string variable.
// outText = std::string(ocr->GetUTF8Text());
// std::cout <<"detected: "<< outText;
// ocr->End();
ocrTOtext(im,outText);
return 0;
}
