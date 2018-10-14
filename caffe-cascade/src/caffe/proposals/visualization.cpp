#include "caffe/proposals/visualization.hpp"
namespace kurff{
void visualize(Mat& img, const vector<Box>& boxes, Scalar scalar){
    LOG(INFO)<<"draw "<< boxes.size()<< " boxes";
    for(auto box : boxes){
        rectangle(img, box, scalar,3);
        putText(img, std::to_string(box.confidence_), Point(box.x,box.y), FONT_HERSHEY_PLAIN,1, Scalar(255,0,0),2);
    }
}
}