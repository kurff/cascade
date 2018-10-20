#ifndef TOOLS_IMAGE_TRANSFORM_HPP__
#define TOOLS_IMAGE_TRANSFORM_HPP__
#include "opencv2/opencv.hpp"
#include "data.hpp"
#include "glog/logging.h"
using namespace cv;
namespace kurff{
    class ImageTransform{
        public:
            ImageTransform(int height, int width): height_(height), width_(width){

            }
            ~ImageTransform(){

            }

            void forward(const cv::Mat & image, cv::Mat& output){
                output = cv::Mat::zeros(height_, width_, CV_8UC3);
                ratio_ = float(image.cols)/float(image.rows);
                scale_ = float(height_) /float(image.rows);
                cv::Mat tmp;
                cv::resize(image, tmp, cv::Size(ratio_*height_, height_));
                for(int i = 0; i < height_; ++ i){
                    for(int j = 0; j < std::min<int>(ratio_*height_, width_); ++ j ){
                        output.at<Vec3b>(i,j) = tmp.at<Vec3b>(i,j);
                    }
                }
            }

            void forward(ImageData& image_data){
                cv::Mat tmp;
                forward(image_data.image, tmp);
                tmp.copyTo(image_data.image);
                for(auto & r : image_data.objects){
                    r.x = float(r.x)*scale_;
                    r.y = float(r.y)*scale_;
                    r.height = float(r.height)*scale_;
                    r.width = float(r.width)*scale_;
                }
            }

            
        protected:
            int height_;
            int width_;
            float ratio_;
            float scale_;




    };






}
#endif