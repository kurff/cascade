#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__
#include "model/model.hpp"
#include "caffe/proposals/utils.hpp"
#include "caffe/proposals/box.hpp"
#include "caffe/proposals/visualization.hpp"
namespace kurff{
    class Classifier : public Model{
        public:
            Classifier(int top_k) : Model(top_k){

            }
            ~Classifier(){

            }
            void run(const Mat& image, vector<Box>& objects ){
                Mat sub;
                vector<float> confidence;
                vector<int> label;
                int height = image.rows;
                int width = image.cols;
                for(auto& obj : objects){
                    sub = image(Rect(obj.x, obj.y, obj.width, obj.height));    
                    run_each(sub, confidence, label);
                }
            }
            
            void run_each(const Mat& image, vector<float>& confidence, vector<int>& label){
                Blob<float>* input_layer = this->net_->input_blobs()[0];
                input_layer->Reshape(1, this->num_channels_, this->input_geometry_.height, this->input_geometry_.width);
                this->net_->Reshape();
                std::vector<cv::Mat> input_channels;
                this->WrapInputLayer(&input_channels);
                this->Preprocess(image, &input_channels);
                this->net_->Forward();
                // Blob<float>* output_layer = this->net_->output_blobs()[0];
                // const float* begin = output_layer->cpu_data();
                // const float* end = begin + output_layer->channels();
                // vector<float> conf(begin, end);
                // vector<int> index = sort_index(conf);

                // confidence.clear();
                // label.clear();
                // for(int i = 0; i < this->top_k_; ++ i){
                //     confidence.push_back(conf[index[i]]);
                //     label.push_back(index[i]);
                // }
            }

            
        protected:
            

        
    };
    CAFFE_REGISTER_CLASS(ModelRegistry, Classifier, Classifier);


}

#endif