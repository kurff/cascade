#include "opencv2/opencv.hpp"
#include <vector>
using namespace cv;
using namespace std;


void split(cv::Mat& gray){
    
    


}



int main(int argc, char* argv[]){
    cv::Mat image = cv::imread(argv[1]);
    cv::Ptr<cv::MSER> mser = cv::MSER::create(2, 10, 5000, 0.5, 0.3);
    std::vector<std::vector<cv::Point> > regContours;
    std::vector<cv::Rect> bboxes;
    cv::Mat gray;
    cv::cvtColor(image, gray, CV_RGB2GRAY);
    mser->detectRegions(gray, regContours, bboxes);

    for(int i = 0; i < bboxes.size(); ++ i){
        cv::rectangle(image, bboxes[i], cv::Scalar(0,0,255) );
    }
    cv::imshow("src", image);
    cv::waitKey(0);
    return 0;
}