#include <algorithm>
#include <vector>

#include "caffe/layers/text_proposal_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TextProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
}

template <typename Dtype>
void TextProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        
    
  
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
    
}


#ifdef CPU_ONLY
STUB_GPU(TextProposalLayer);
#endif

INSTANTIATE_CLASS(TextProposalLayer);

}  // namespace caffe
