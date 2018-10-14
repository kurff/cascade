#include <algorithm>
#include <vector>

#include "caffe/layers/text_proposal_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proposals/CannyProposal.hpp"

namespace caffe {

template<typename Dtype>
void TextProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    TextProposalParameter text_proposal_param = this->layer_param_.text_proposal_param();
    num_proposals_ = text_proposal_param.num_proposals();
    min_size_ = text_proposal_param.min_size();
    proposals_.clear();
    for(int i = 0; i < text_proposal_param.proposal_method_size(); ++ i){
        proposals_.push_back(kurff::ProposalRegistry()->Create(text_proposal_param.proposal_method(i)+"Proposal", min_size_));
    }
    //transform_param_(text_proposal_param.transform_param());


    
}



template <typename Dtype>
void TextProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Nx1x1x6 [index of batch, x0, y0, x1, y1, confidence];
    top[0]->Reshape(vector<int>{num_proposals_, 1, 1, 6});
}

template <typename Dtype>
void TextProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();
    const Dtype* data = bottom[0]->cpu_data();
    int channel = bottom[0]->channels();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int top_dim = top[0]->count(1);
    int bottom_dim = bottom[0]->count(1);
    for(int i = 0; i < num; ++ i){
        vector<kurff::Box> boxes;
        for(int j = 0; j < proposals_.size(); ++ j){
            vector<kurff::Box> proposal;
            proposals_[j]->run<Dtype>(data + i*bottom_dim, height, width, channel, proposal);
            boxes.insert(boxes.end(), proposal.begin(), proposal.end());
        }
        std::sort(boxes.begin(), boxes.end(), kurff::compare);
        
        // sort boxes proposals, return top N candidate
        for(int i = 0; i < std::min<int>(boxes.size(), num_proposals_); ++ i){
            top_data[i*6+0] = i;
            top_data[i*6+1] = boxes[i].x;
            top_data[i*6+2] = boxes[i].y;
            top_data[i*6+3] = boxes[i].x + boxes[i].width;
            top_data[i*6+4] = boxes[i].y + boxes[i].height;
            top_data[i*6+5] = boxes[i].confidence_;
            top_data += top_dim;
        }
    }
}

template <typename Dtype>
void TextProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}




INSTANTIATE_CLASS(TextProposalLayer);

REGISTER_LAYER_CLASS(TextProposal);
}  // namespace caffe
