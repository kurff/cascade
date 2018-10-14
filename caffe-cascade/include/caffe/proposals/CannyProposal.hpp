#ifndef __CANNY_PROPOSAL__
#define __CANNY_PROPOSAL__

#include "opencv2/opencv.hpp"
#include "caffe/proposals/Proposal.hpp"
#include "caffe/proposals/utils.hpp"

#include <memory>
namespace kurff{
    class CannyProposal : public Proposal{
        public:
            CannyProposal(int min_size, int ratio = 3, int lowThreshold = 20, int kernel_size = 3): Proposal(min_size)
            , ratio_(ratio), lowThreshold_(lowThreshold), kernel_size_(kernel_size), ratio_size_(1.2f){
                this->name_="Canny";
            }
            ~CannyProposal(){

            }

            void run(const Mat& gray, vector<Box>& proposals);
        protected:
            int ratio_;
            int lowThreshold_;
            int kernel_size_;
            float ratio_size_;
            


    };

}




#endif