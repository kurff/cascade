#ifndef __PROPOSAL_HPP__
#define __PROPOSAL_HPP__
#include "opencv2/opencv.hpp"
#include "caffe/proposals/box.hpp"
#include "caffe/proposals/registry.hpp"
#include "glog/logging.h"
using namespace cv;
namespace kurff{
    class Proposal{
        public:
            Proposal(){

            }
            ~Proposal(){

            }
         
            string name();
            virtual void run(const Mat& image, vector<Box>& proposals) = 0;
            template<typename Dtype>
            void run(const Dtype* data, int height, int width, int channel, vector<Box>& proposals);
            virtual void evaluate(const Mat& image, vector<Box>& proposals);

        protected:
            string name_;
    };
    CAFFE_DECLARE_REGISTRY(ProposalRegistry, Proposal);

}

#endif