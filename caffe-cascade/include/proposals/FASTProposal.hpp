#ifndef __FAST_PROPOSAL_HPP__
#define __FAST_PROPOSAL_HPP__
#include <vector>
#include "glog/logging.h"
#include "proposals/Proposal.hpp"
#include "proposals/fast/fast.h"
namespace kurff{
    class FASTProposal : public Proposal{
        public:

            FASTProposal():Proposal(){
                this->name_="FAST";
            }
            ~FASTProposal(){

            }

            void run(const Mat& image, vector<Box>& proposals){


            }

        protected:


    

    };
    CAFFE_REGISTER_CLASS(ProposalRegistry, FASTProposal, FASTProposal);



}
#endif