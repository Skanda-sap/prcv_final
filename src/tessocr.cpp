#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include "/home/sid/Documents/prcv projects/tocr/include/helper.h"

int main(){ 

//load images
std::string outText, imPath = "/home/sid/Documents/prcv projects/tocr/img/ma.png";
cv:: Mat im = cv::imread(imPath, cv::IMREAD_COLOR);

ocrTOtext(im,outText);
return 0;
}
