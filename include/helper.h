#ifndef HELPER_H
#define HELPER_H

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>

int ocrTOtext(cv::Mat& im, std::string& outText);

#endif

