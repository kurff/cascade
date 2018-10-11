#ifndef __MSERPROPOSAL_HPP__
#define __MSERPROPOSAL_HPP__

#include <memory>
#include "proposals/mser/mser.h"
#include "proposals/Proposal.hpp"
#include "proposals/Merge.hpp"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "utils/utils.hpp"

namespace kurff{    
    class MSERProposal: public Proposal{
        public:
            MSERProposal(int number_proposals): Proposal(number_proposals){
                this->name_ = "MSER";

            }
            ~MSERProposal(){

                
            }

            void run(const Mat& image, vector<Box>& proposals){
        
                //LOG(INFO)<<"8: "<< proposals.size();
                
            }

        protected:
            
            



    };
    CAFFE_REGISTER_CLASS(ProposalRegistry, MSERProposal, MSERProposal);

}
#endif