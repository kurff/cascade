#include "caffe/proposals/visualization.hpp"
namespace kurff{
void visualize(Mat& img, const vector<Box>& boxes, Scalar scalar){
    LOG(INFO)<<"draw "<< boxes.size()<< " boxes";
    for(auto box : boxes){
        rectangle(img, box, scalar,3);
    }
}
}