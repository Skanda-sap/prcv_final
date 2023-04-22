// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
// Yolo V5 requires input image of this dimension
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
// const float SCORE_THRESHOLD = 0.5;
const float SCORE_THRESHOLD = 0.25;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.25;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

const int nc = 1; // Number of classes

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);


// Draw the predicted bounding box.
void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    std::cout << "Baseline: " << baseLine << std::endl;
    top = max(top, label_size.height);
    // Top left corner.
    // Point tlc = Point(left, top);
    Point tlc = Point(left, top - baseLine);
    // Bottom right corner.
    // Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    Point brc = Point(left + label_size.width, top - label_size.height - 2*baseLine);
    // Draw black rectangle.

    // Compute the position of the text within the bounding box
    cv::Point textPosition(left, top - label_size.height);

    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    // putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
    putText(input_image, label, textPosition, FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    std::cout << "Input image is: " << input_image.type() << std::endl;
    Mat blob;
    std::cout << "I/P image dims: " << input_image.dims << std::endl;
    // blobFromImage converts image from CxWxH to 1xCxINPUT_WIDTHxINPUT_HEIGHT or a tensor. subtraction, division by the given argument.. if its 1/255, O/P image lies between [0,1] for each pixel 
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    // imshow("blob_image",input_image);
    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;
    Mat blob_reshaped = blob.reshape(1,1);
    std::cout << "Blob reshaped: " << blob_reshaped.dims << std::endl;

    minMaxLoc( blob_reshaped, &minVal, &maxVal, &minLoc, &maxLoc );

    cout << "min val in blob: " << minVal << " at location " << minLoc << endl;
    cout << "max val in blob: " << maxVal << " at location " << maxLoc << endl;    
    std::cout << "Blob type is: " << blob.type() << std::endl;

    // Blob's mat type is 5. i.e. CV_32FC1
    net.setInput(blob);
    // net.setInput(input_image);

    // Forward propagate.
    std::vector<cv::String> outputNames = net.getUnconnectedOutLayersNames();
    std::cout << "getUnconnectedOutLayersNames size: " << outputNames.size() << std::endl;
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    std::cout << "Preprocessing done: outputs size: " << outputs[0].size << std::endl;
    return outputs;
}


Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name) 
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes; 

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    // std::cout << outputs[0].size << std::endl;
    // std::cout << outputs[0].size() << std::endl;
    // std::cout << outputs[0].size().width << std::endl;
    // std::cout << outputs[0].size().height << std::endl;
    
    float *data = (float *)outputs[0].data;
    std::cout << data << std::endl;

    const int dimensions = nc + 5;
    // const int dimensions = 85;
    const int rows = 25200;  // Remains fixed
    // std::cout << outputs[0].rows << std::endl;
    // const int rows = outputs[0].size.cols;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5; // Point to the first class
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores); // The matrix data is initialized with the values from the array classes_scores, which is a pointer to a block of memory containing the matrix data.

            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                // std::cout << "classid x: " << class_id.x << " | class id y: " << class_id.y << std::endl;
                // std::cout << "classid x: " << class_id.x << " | class id y: " << class_id.y << std::endl;
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
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        // data += 85;
        data += nc + 5;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) 
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}


int main()
{
    // Load class list.
    vector<string> class_list;
    // ifstream ifs("coco.names");
    ifstream ifs("number_plate.names");
    
    string line;

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    std::cout << "Number of classes: " << class_list.size() << std::endl;

    // Load image.
    Mat frame;
    // frame = imread("sample.jpg");
    frame = imread("MicrosoftTeams-image.png");
    // std::cout << "I/P image dims: " << frame.channels() << std::endl;
    // cout << "OpenCV version : " << CV_VERSION << endl;
    // Load model.
    Net net;
    net = readNet("models/best_include_torchscript.onnx"); 
    // net = readNet("models/dev_yolov5s_w_op.onnx"); // WORKS
    
    
    // net = readNet("models/best_w_simplify_op.onnx"); // WORKS
    
    // net = readNet("models/yolov5s.onnx"); 

    vector<Mat> detections;
    detections = pre_process(frame, net);
    std::cout << "Detections from pre_process size: " << detections[0].size << std::endl;

    cv::Mat frame_cloned = frame.clone();
    Mat img = post_process(frame_cloned, detections, class_list);

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time : %.2f ms", t);
    putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    imshow("Output", img);
    waitKey(0);

    return 0;
}