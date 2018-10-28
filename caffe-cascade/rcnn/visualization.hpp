#ifndef RCNN_VISUALIZATION_HPP__
#define RCNN_VISUALIZATION_HPP__
#include <vector>
#include "opencv2/opencv.hpp"
#include "caffe/util/bbox_util.hpp"
using namespace caffe;
using namespace std;
using namespace cv;
namespace rcnn{
    void visualize_faster_rcnn(cv::Mat & image, vector<BBox>& boxes, Scalar scalar){
        for(auto b : boxes){
            cv::rectangle(image, Rect2f(b.xmin, b.ymin, b.xmax-b.xmin, b.ymax-b.ymin), scalar, 2);
            cv::putText(image, std::to_string(b.score), Point(b.xmin-10,b.ymin-10), FONT_HERSHEY_PLAIN,2, scalar,2);
        }
    }
}


#endif