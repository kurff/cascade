#ifndef TOOLS_IMAGE_TRANSFORM_HPP__
#define TOOLS_IMAGE_TRANSFORM_HPP__
#include "opencv2/opencv.hpp"
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
                float ratio = float(image.cols)/float(image.rows);
                cv::Mat tmp;
                cv::resize(image, tmp, cv::Size(ratio*height_, height_));

                for(int i = 0; i < height_; ++ i){
                    for(int j = 0; j < std::min<int>(ratio*height_, width_); ++ j ){
                        output.at<Vec3b>(i,j) = tmp.at<Vec3b>(i,j);
                    }
                }
            }

            
        protected:
            int height_;
            int width_;




    };






}
#endif