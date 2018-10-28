#ifndef __MASK_RCNN_H__
#define __MASK_RCNN_H__
#include "rcnn/fast_rcnn.h"
namespace rcnn{
    class MaskRCNN : public FastRCNN{
        public:
            MaskRCNN(float threshold) : FastRCNN(threshold){

            }
            ~MaskRCNN(){

            }
        
        public:
      
             bool forward(cv::Mat& image, std::vector<BBox>* boxes);

             bool forward(cv::Mat& image, const std::vector<BBox>& prev_boxes, std::vector<BBox>* curr_boxes);

        protected:


    };



}


#endif