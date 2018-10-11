#ifndef __PROPOSAL_HPP__
#define __PROPOSAL_HPP__
#include "opencv2/opencv.hpp"
#include "proposals/box.hpp"
#include "proposals/registry.h"
#include "glog/logging.h"
using namespace cv;
namespace kurff{
    class Proposal{
        public:
            Proposal(){

            }
            ~Proposal(){

            }
         
            string name(){return name_;}
            virtual void run(const Mat& image, vector<Box>& proposals) = 0;
            template<typename Dtype>
            void run(const Dtype* data, int height, int width, int channel, vector<Box>& proposals){
                cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
                for(int i = 0; i < height; ++ i){
                    for(int j = 0; j < width; ++ j){
                        uchar b = uchar(data[i*3*width + j *3]);
                        uchar g = uchar(data[i*3*width + j*3 + 1]);
                        uchar r = uchar(data[i*3*width + j*3 + 2]);
                        image.at<Vec3b>(i,j) = Vec3b(b,g,r);
                    }
                }
                cv::Mat gray;
                cvtColor( image, gray, CV_BGR2GRAY ); 
                run(gray, proposals);
                evaluate(gray, proposals);
            }
            virtual void evaluate(const Mat& image, vector<Box>& proposals){
                for(int i = 0; i < proposals.size(); ++ i){
                    Mat sub = image(proposals[i]);
                    Mat dst;
                    equalizeHist(sub,dst);
                    Mat m, d;
                    meanStdDev(sub, m, d);
                    proposals[i].confidence_ = d.at<double>(0,0);
                }
            }

        protected:
            string name_;
    };
    CAFFE_DECLARE_REGISTRY(ProposalRegistry, Proposal);
    CAFFE_DEFINE_REGISTRY(ProposalRegistry, Proposal);





}

#endif