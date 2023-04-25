#include <opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<math.h>

using namespace std;
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

cv::Mat createMask(const cv::Mat &myImage, float width, float height)
{
    cv::Mat mask = cv::Mat::zeros(myImage.size(), myImage.type());
    cv::Point pts[1][4];
    pts[0][0] = cv::Point(width * 0.30, height);
    pts[0][1] = cv::Point(width * 0.46, height * 0.72);
    pts[0][2] = cv::Point(width * 0.58, height * 0.72);
    pts[0][3] = cv::Point(width * 0.82, height);
    const cv::Point* ppt[1] = { pts[0] };
    int npt[] = { 4 };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255,255,255));
    return mask;
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
    int histSize = 256;  // Number of bins
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::calcHist(&frame, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange);

    // Find mid point on histogram
    int midpoint = histogram.cols / 2;
    std::cout<<"MidPoint: "<<midpoint<<std::endl;

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

    if (left_x_base == 0 || right_x_base == frame.cols) {
        std::cout << "Lane not detected." << std::endl;
    }

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
        // double x_diff = x2 - x1;
        // double slope = (y2 - y1) / x_diff;
        // if (x_diff < 0) {
        //     slope = -slope;
        // }
        // double slope = (y2 - y1) / (x2 - x1) * std::sign(x2 - x1);
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
        // lane_lines.push_back(map_coordinates(frame, left_fit_average));
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
        // lane_lines.push_back(map_coordinates(frame, right_fit_average));
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
        std::cout<<"up mid: "<<up_mid<<std::endl;
        std::cout<<"low mid: "<<low_mid<<std::endl;
       
        return {up_mid, low_mid};
    }
}

cv::Mat add_text(cv::Mat frame, int image_center, int left_x_base, int right_x_base) {
    std::cout<<" lef base : "<<left_x_base<<std::endl;
    std::cout<<" rigt base : "<<right_x_base<<std::endl;

    double lane_center = left_x_base + (right_x_base - left_x_base) / 2;
    std::cout<<" lane center : "<<lane_center<<std::endl;
    double deviation = image_center - lane_center;
    std::cout<<" deviation: "<<deviation<<std::endl;
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


int main(int argc, char* argv[]) {
    cv::Mat myImage;
    cv::Mat grayVideo;
    cv::Mat cannyedgeVideo;
    cv::namedWindow("Video Player");

    cv::VideoCapture cap("/home/skanda/Documents/prcv_final/input/input_video5.mp4");
    int width = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Video size: " << width << " x " << height << std::endl;


    if (!cap.isOpened()) {
        std::cout << "Unable to open video file" << std::endl;
        return -1;
    }

    cv::Mat mask;
    while (true) {
        cap >> myImage;

        if (myImage.empty()) {
            break;
        }

        char c = (char) cv::waitKey(10);

        if (c == 'q' || c == 'Q') {
            break;
        }

        imshow("Video Player", myImage);
// denoising
        cv::Mat blur_video = applyGaussianBlur(myImage);
        // cv::imshow("Gaussian blur video", blur_video);
// grayscale of denoised frame
        grayVideo = convertToGrayscale(blur_video);
        // cv::imshow("gray_scale_after_blur", grayVideo);
// canny edge detection
        cannyedgeVideo = applyCanny(grayVideo, 50, 3, 3);
        cv::imshow("cannyVideo", cannyedgeVideo);        
// fill poly --> masking
        // createMask(mask, cannyedgeVideo, width, height);
        cv::Mat mask = createMask(cannyedgeVideo, width, height);
        cv::imshow("mask", mask);
        // std::cout << "Width of mask: " << mask.cols << ", Height of mask: " << mask.rows << std::endl;
        // cv::Mat cropped_edges = mask.clone();
        cv::Mat cropped_edges = cv::Mat::zeros(mask.size(), mask.type());
        cv::bitwise_and(grayVideo, mask, cropped_edges);
        cv::imshow("crop edge Video", cropped_edges);

        cv::Mat canny_masked = cv::Mat::zeros(mask.size(), mask.type());
        cv::bitwise_and(cannyedgeVideo, mask, canny_masked);
        cv::imshow("canny masked Video", canny_masked);

        warpPerspective(cannyedgeVideo);
        // cv::imshow("after warp cannyVideo", cannyedgeVideo);        

        cv::Mat warped_image = myImage.clone();  // Make a copy of the input image

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
    // cv::line(myImage, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
    }
    // std::cout<<"line segment completed"<<std::endl;
    // cv::imshow("Hough",myImage);

            // cv::imshow("detected_lines", warped_image_gray); // display the image with the detected
         std::vector<cv::Vec4i> optimized_lines = optimize_lines(cannyedgeVideo, line_segments);
        // std::vector<cv::Vec4i> optimized_lines = optimize_lines(myImage, line_segments);
        cv::Mat outputFrame = display_lines(myImage, optimized_lines);
        std::pair<int, int> shifting_points = get_floating_center(outputFrame, optimized_lines);
        int imageCenter = outputFrame.cols / 2;
        cv::Mat frameWithText = add_text(outputFrame, shifting_points.first, left_x_base, right_x_base);
        cv::imshow("Frame with Text", frameWithText);

/*

        */
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}