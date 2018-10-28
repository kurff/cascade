#include "rcnn/fast_rcnn.h"
#include "rcnn/faster_rcnn.h"
#include "rcnn/visualization.hpp"
using namespace rcnn;
int main(){
    FastRCNN* faster_rcnn = new FasterRCNN(0.2); 
    string proto = "/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/cascade/caffe-cascade/examples/coco/res50-15s-800-fpn-base-pretrained/deploy.prototxt";
    string model = "/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/cascade/caffe-cascade/examples/coco/res50-15s-800-fpn-base-pretrained/cascadercnn_coco_iter_280000.caffemodel";
    faster_rcnn->init(proto, model , -1);
    cv::Mat img = cv::imread("fish-bike.jpg");
    std::vector<BBox> boxes;
    faster_rcnn->forward(img, & boxes);

    LOG(INFO)<<"box size: "<< boxes.size();
    for(auto b : boxes){
        LOG(INFO)<<"conf: "<< b.score<<" "<< b.xmin <<" "<<b.ymin<<" "<< b.xmax<<" "<< b.ymax;
    }

    rcnn::visualize_faster_rcnn(img, boxes, Scalar(0,0,255));

    cv::imshow("result", img);

    cv::waitKey(0);

    return 0;
}