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
    cache_.clear();
    int number = bottom[0]->num();
    for(int i = 0; i < number; ++ i){
        std::shared_ptr<Blob<Dtype> > ptr(new Blob<Dtype>());
        cache_.push_back(ptr);
    }
}



template <typename Dtype>
void TextProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Nx1x1x6 [index of batch, x0, y0, x1, y1, confidence];
    //top[0]->Reshape(vector<int>{num_proposals_, 1, 1, 5});
}

template <typename Dtype>
void TextProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();
    const Dtype* data = bottom[0]->cpu_data();
    int channel = bottom[0]->channels();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int bottom_dim = bottom[0]->count(1);
    int total_num = 0;
    for(int i = 0; i < num; ++ i){
        vector<kurff::Box> boxes;
        for(int j = 0; j < proposals_.size(); ++ j){
            vector<kurff::Box> proposal;
            proposals_[j]->run<Dtype>(data + i*bottom_dim, height, width, channel, proposal);
            boxes.insert(boxes.end(), proposal.begin(), proposal.end());
        }
        std::sort(boxes.begin(), boxes.end(), kurff::compare);

        // sort boxes proposals, return top N candidate
        int num_candidate = std::min<int>(boxes.size(),num_proposals_);

        cache_[i]->Reshape(vector<int>{num_candidate, 1, 1, 5});
        int dim = cache_[i]->count(1);
        Dtype* cache_data = cache_[i]->mutable_cpu_data();

        for(int j = 0; j < num_candidate; ++ j){
            cache_data[0] = i;
            cache_data[1] = boxes[j].x;
            cache_data[2] = boxes[j].y;
            cache_data[3] = boxes[j].x + boxes[j].width;
            cache_data[4] = boxes[j].y + boxes[j].height;
            //top_data[i*6+5] = boxes[i].confidence_;
            cache_data += dim;
        }
        total_num += num_candidate;
    }
    top[0]->Reshape(vector<int>{total_num, 1, 1 ,5});
    Dtype* top_data = top[0]->mutable_cpu_data();
    int top_dim = top[0]->count(1);
    for(int i = 0; i < num; ++ i){
        for(int j = 0; j < cache_[i]->num(); ++ j){
            top_data[0] = cache_[i]->data_at(j,0,0,0);
            top_data[1] = cache_[i]->data_at(j,0,0,1);
            top_data[2] = cache_[i]->data_at(j,0,0,2);
            top_data[3] = cache_[i]->data_at(j,0,0,3);
            top_data[4] = cache_[i]->data_at(j,0,0,4);
        }
        top_data += top_dim;
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
