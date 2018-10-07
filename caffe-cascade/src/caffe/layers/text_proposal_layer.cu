#include <algorithm>
#include <vector>

#include "caffe/layers/text_proposal_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void TextProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        
    
  
}

template <typename Dtype>
void TextProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
    
}



INSTANTIATE_LAYER_GPU_FUNCS(TextProposalLayer);

}  // namespace caffe
