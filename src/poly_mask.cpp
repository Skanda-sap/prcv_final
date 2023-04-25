#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

// Global variables
vector<Point> vertices;
Mat mask; // moved the mask initialization outside the selectROI function

cv::Mat createMask(const cv::Mat &myImage, const vector<Point>& vertices) {
    cv::Mat mask = cv::Mat::zeros(myImage.size(), myImage.type());
    const cv::Point* ppt[1] = { &vertices[0] };
    int npt[] = { static_cast<int>(vertices.size()) };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255,255,255));
    return mask;
}
void savePointsToFile(const vector<Point>& vertices, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return;
    }

    for (const auto& vertex : vertices) {
        outFile << vertex.x << " " << vertex.y << endl;
    }

    outFile.close();
    cout << "Points saved to file: " << filename << endl;
}

// Callback function for mouse events
void selectROI(int event, int x, int y, int flags, void* userdata) {
    Mat frame = *((Mat*)userdata);

    if (event == EVENT_LBUTTONDOWN) {
        vertices.push_back(Point(x, y));
        circle(frame, Point(x, y), 5, Scalar(0, 0, 255), -1);

        if (vertices.size() > 1) {
            vector<vector<Point>> pts{ vertices };
            mask = createMask(frame, vertices);
            // cout << "Updated mask:\n" << mask << endl; // print the updated mask
        }

        imshow("Select ROI", frame);
        } else if (event == EVENT_RBUTTONDOWN) {
            if (vertices.size() > 0) {
                savePointsToFile(vertices, "home/skanda/Documents/prcv_final/input/roi_points.txt");
            }
            vertices.clear();
            mask = Mat::zeros(frame.size(), CV_8UC1);
            imshow("Select ROI", frame);
        }

    // Print the current mask
    // cout << "Mask:\n" << mask << endl;
}

int main(int argc, char** argv) {
    // Open the video file
    VideoCapture cap("/home/skanda/Documents/prcv_final/input/inputVideo.mp4");
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return -1;
    }

    // Create the GUI window
    namedWindow("Select ROI", WINDOW_NORMAL);

    // Read the sixth frame from the video
    Mat frame, sixth_frame;
    for (int i = 0; i < 6; i++) {
        cap.read(frame);
    }
    sixth_frame = frame.clone();

    // Display the sixth frame and wait for the user to select the ROI
    imshow("Select ROI", sixth_frame);
    setMouseCallback("Select ROI", selectROI, &sixth_frame);
    while (vertices.size() < 4) {
        waitKey(1);
    }

    // Create the mask region that corresponds to the ROI
    Mat mask = createMask(sixth_frame, vertices);

while (cap.isOpened()) {
    // Read the next frame from the video
    cap.read(frame);
    if (frame.empty()) {
        break;
    }

    // Apply the mask to the current frame
    Mat masked_frame;
    frame.copyTo(masked_frame, mask);

    // Display the masked frame
    imshow("Masked Frame", masked_frame);

    // Exit if the user presses the ESC key
    if (waitKey(1) == 27) {
        break;
    }
}

// Release the video capture object and close all windows
cap.release();
destroyAllWindows();

return 0;
}