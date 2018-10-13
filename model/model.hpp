#ifndef __MODEL_HPP__
#define __MODEL_HPP__
#include <memory>
#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"
#include "caffe/proposals/registry.hpp"
#include "caffe/proposals/box.hpp"

#include <string>
using namespace std;
using namespace caffe;
using namespace cv;
namespace kurff{
    class Model{
        public:
            Model(int top_k): top_k_(top_k){

            }
            ~Model(){

            }

            void init(const string& proto, const string& weight_file, bool use_gpu){
                if(use_gpu){
                    Caffe::set_mode(Caffe::GPU);
                }else{
                    Caffe::set_mode(Caffe::CPU);
                }
                
                net_.reset(new Net<float>(proto, TEST));
                //LOG(INFO)<<"-----------------------------------------";
                if(!weight_file.empty()){
                    net_->CopyTrainedLayersFrom(weight_file);
                }
                
                
                Blob<float>* input_layer = net_->input_blobs()[0];
                num_channels_ = input_layer->channels();
                CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
                input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
                SetMean();
            }

            //template<typename T>
            virtual void run(const Mat& image, vector<Box>& objects ) = 0;

            virtual void run_each(const Mat& image, vector<float>& confidence, vector<int>& label) = 0;

            void WrapInputLayer(std::vector<cv::Mat>* input_channels) {
                Blob<float>* input_layer = net_->input_blobs()[0];

                int width = input_layer->width();
                int height = input_layer->height();
                float* input_data = input_layer->mutable_cpu_data();
                for (int i = 0; i < input_layer->channels(); ++i) {
                    cv::Mat channel(height, width, CV_32FC1, input_data);
                    input_channels->push_back(channel);
                    input_data += width * height;
                }
            } 

            void SetMean(){
                vector<float> values ={104,117,123};
                std::vector<cv::Mat> channels;
                for (int i = 0; i < num_channels_; ++i) {
                    cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                    cv::Scalar(values[i]));
                    channels.push_back(channel);
                }
                cv::merge(channels, mean_);
            }
            void Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
                cv::Mat sample;
                if (img.channels() == 3 && num_channels_ == 1)
                    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
                else if (img.channels() == 4 && num_channels_ == 1)
                    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
                else if (img.channels() == 4 && num_channels_ == 3)
                    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
                else if (img.channels() == 1 && num_channels_ == 3)
                    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
                else
                    sample = img;

                cv::Mat sample_resized;
                if (sample.size() != input_geometry_)
                    cv::resize(sample, sample_resized, input_geometry_);
                else
                    sample_resized = sample;

                cv::Mat sample_float;
                if (num_channels_ == 3)
                    sample_resized.convertTo(sample_float, CV_32FC3);
                else
                    sample_resized.convertTo(sample_float, CV_32FC1);

                cv::Mat sample_normalized;
                //cv::subtract(sample_float, mean_, sample_normalized);
                cv::split(sample_float, *input_channels);

                //CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
                //<< "Input channels are not wrapping the input layer of the network.";
            }
        protected:
            std::shared_ptr<Net<float> > net_;
            cv::Size input_geometry_;
            int num_channels_;
            cv::Mat mean_;
            int top_k_;

    };
    CAFFE_DECLARE_REGISTRY(ModelRegistry, Model, int);
    CAFFE_DEFINE_REGISTRY(ModelRegistry, Model, int);


}

#endif