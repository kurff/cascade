#ifndef __FASTER_RCNN_H__
#define __FASTER_RCNN_H__
#include "rcnn/fast_rcnn.h"
#include <memory>
namespace rcnn{
    class FasterRCNN : public FastRCNN{
        public:
            FasterRCNN(float threshold);
            ~FasterRCNN();

  
             bool forward(cv::Mat& image, std::vector<BBox>* boxes);

             bool forward(cv::Mat& image, const std::vector<BBox>& prev_boxes, std::vector<BBox>* curr_boxes);



        protected:



    };




}


#endif