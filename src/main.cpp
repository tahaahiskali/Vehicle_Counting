#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <getopt.h>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "CentroidTracker.h"

using namespace cv;
using namespace std;

void print_usage();

int main( int argc, char **argv ) {

    if( argc < 5 ) {
        print_usage();
    }

    struct option long_options[] =
    {
        {"input",			required_argument, 0, 'i'},
        {"output",			required_argument, 0, 'o'},
        {"disable_gui",		optional_argument, 0, 'g'},
    };

    std::string inputFileName;
    std::string outputFileName;
	bool enable_gui = true;

    ofstream streamFile;

    int c;

    while (true) {

        int option_index = 0;

        c = getopt_long( argc, argv, "i:o:g", long_options, &option_index);

        if( c == -1 ) {
            break;
        }

        if( c != 'i' && c != 'o' && c!= 'g' ) {
            print_usage();
        }

        switch (c) {
        case 'i':
            inputFileName = optarg;
            break;
        case 'o':
            outputFileName = optarg;
            break;
        case 'g':
            enable_gui = false;
            break;
        default:
            break;
        }
    }

    auto centroidTracker = new CentroidTracker(8);

    VideoCapture cap(inputFileName);
	int length = int(cap.get(CAP_PROP_FRAME_COUNT));

    if (!cap.isOpened()) {
        cout << "Cannot open the input file" << endl;
        exit(1);
    }

    String modelTxt = "../model/vehicle-detection-0202/vehicle-detection-0202.xml";
    String modelBin = "../model/vehicle-detection-0202/vehicle-detection-0202.bin";

    cout << "Loading model" << endl;
    auto net = dnn::readNet(modelTxt, modelBin);

    std::string ID;
    std::string previousID;

    bool handled = false;
    struct timeval now;
    struct timeval prev;

    memset(&now,0,sizeof(now));
    memset(&prev,0,sizeof(prev));

    double average = 0;
	int frame_counter = 0;

    cout << "Starting video stream" << endl;

    while (cap.isOpened()) {

		++frame_counter;
        Mat frame;
        handled = cap.read(frame);

		std::cout << " Processing " << std::setw(5) << frame_counter << ". frame" << std::setw(2) << " ["<< length << "]" << std::endl;

        if( handled != true) {
            break;
        }

        // Calculate Average frame processing time
        gettimeofday(&now, NULL);
        average = 1 / ( (double)(now.tv_usec - prev.tv_usec) / 1000000 + (double)(now.tv_sec - prev.tv_sec));
        prev = now;

        // resize frame
        resize(frame, frame, Size(1024, 768));

        auto inputBlob = dnn::blobFromImage(frame, 1.0, Size(512, 512),true,CV_32FC3);
        net.setInput(inputBlob);
        auto detection = net.forward();
        Mat detectionMat(200, 7, CV_32F, detection.ptr<float>());

        vector<vector<int>> boxes;

        for (int i = 0; i < detectionMat.rows; i++) {
            int label = static_cast<int>(detectionMat.at<int>(i, 1));
            float confidence = static_cast<float>(detectionMat.at<float>(i, 2));
            int xLeftTop = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftTop = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightBottom = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightBottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            if( confidence > 0.4 && label == 0 ) {
                Rect object((int) xLeftTop, (int) yLeftTop, (int) (xRightBottom - xLeftTop),
                            (int) (yRightBottom - yLeftTop));
                rectangle(frame, object, Scalar(255, 0, 0));
                boxes.insert(boxes.end(), {xLeftTop, yLeftTop, xRightBottom, yRightBottom});
            }
        }

        auto objects = centroidTracker->update(boxes);

        if (!objects.empty()) {
            for (auto obj: objects) {
                ID = to_string(obj.first+1);
                if( atoi(ID.c_str())+10 < atoi(previousID.c_str()) ) continue;
                previousID = ID;
                std::string text = "vehicle_" + ID;
                cv::putText(frame, text, Point(obj.second.first - 10, obj.second.second - 10),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                circle(frame, Point(obj.second.first, obj.second.second), 2, Scalar(0, 255, 200),1);
            }
        }

        string estTxt = "Estimated Vehicles: " + previousID;
        cv::putText(frame, estTxt, Point(20,520),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 255, 200), 2);

        string fpsTxt = "Avg. Frame Processing Time: " + to_string(average);
        cv::putText(frame, fpsTxt, Point(20,560),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 255, 200), 2);

		if( enable_gui ) {
			namedWindow("cars", WINDOW_AUTOSIZE);
			imshow("cars", frame);
		}

        char c = (char) waitKey(15);
        if (c == 27)
            break;
    }

	cout << "Ending video stream" << endl;

    streamFile.open(outputFileName, std::ios::out);
    streamFile << "Vehicle Count: " << atoi(previousID.c_str());
    streamFile << endl;
    streamFile << "Average frame processing time: " << average;
    streamFile << endl;
    streamFile.close();

    delete centroidTracker;
    destroyAllWindows();
    cap.release();
    return 0;
}

void print_usage()
{
	std::cout << "usage:\n\t";
	std::cout << std::flush;
    fprintf(stderr, "./VehicleCounter --input example_vehicle_video.mp4 --output result.txt --disable_gui");
	std::cout << "\n\t" << std::flush;
    fprintf(stderr, "./VehicleCounter --input example_vehicle_video.mp4 --output result.txt\n");
    exit(EXIT_FAILURE);
}

