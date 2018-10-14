#include <algorithm>
#include <vector>

#include "caffe/layers/transform_data_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template<typename Dtype>
void TransformDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    transform_param_ = this->layer_param_.transform_param();
    data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, TEST));
    data_transformer_->InitRand();    
}



template <typename Dtype>
void TransformDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void TransformDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    data_transformer_->Transform(bottom[0], top[0]);
    

}

template <typename Dtype>
void TransformDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}




INSTANTIATE_CLASS(TransformDataLayer);

REGISTER_LAYER_CLASS(TransformData);
}  // namespace caffe
