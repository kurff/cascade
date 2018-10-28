#ifndef __FAST_RCNN_H__
#define __FAST_RCNN_H__
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/caffe.hpp"
using namespace caffe;
using namespace cv;

namespace rcnn{
    class FastRCNN{
        public:
            FastRCNN(float threshold);
            ~FastRCNN();


            bool init(const std::string& prototxt, const std::string& caffemodel, const int gpu_id);

            // input 
            virtual bool forward(cv::Mat& image, std::vector<BBox>* boxes) = 0;

            virtual bool forward(cv::Mat& image, const std::vector<BBox>& prev_boxes, std::vector<BBox>* curr_boxes) = 0;

        protected:

            void WrapInputLayer(std::vector<cv::Mat>* input_channels);
            void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

        protected:
            std::shared_ptr<caffe::Net<float> > net_;
            cv::Size input_geometry_;
            float shortSize_;
            float longSize_;
            int num_channels_;
            float threshold_;


    };


}// end of namespace rcnn



#endif