#ifndef CASCADE_RANDOM_FOREST_DATA_HPP_
#define CASCADE_RANDOM_FOREST_DATA_HPP_
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <assert.h>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include <cmath>


using namespace cv;
using namespace std;
namespace kurff{
    typedef struct Element_{
        
        vector<float> feat_;
        int number_;
        float v_;
        string label_name_;
        int index_;


    }Element;


    bool compare(const std::shared_ptr<Element>& e1, const std::shared_ptr<Element>& e2){
        return e1->v_ < e2->v_;
    }

    class Data{
        public:
            Data():height_(100), width_(100){
                

            }

            ~Data(){


            }

            void load_image_data(string file){
                ifstream fi(file.c_str(), ios::in);
                string name;
                string image_name;
                int number;
                int cnt = 0;
                while(fi >> image_name){
                    cv::Mat image = cv::imread(image_name);
                    if(cnt ==0){
                        height_ = image.rows;
                        width_ = image.cols;
                    }
                    ++ cnt;
                    std::shared_ptr<Element> feat(new Element());
                    feat->feat_.clear();
                    assert(image.channels() == 1);
                    assert(image.rows == height_ && image.cols == width_);
                    for(int i = 0; i < image.rows; ++ i){
                        for(int j = 0; j < image.cols; ++ j){
                            uchar v = image.at<uchar>(i,j);
                            feat->feat_.push_back(float(v)/ 255.0f);
                        }
                    }
                    feat->number_ = 1;
                    feat->label_name_ = std::to_string(cnt);
                    //feat_.insert(std::make_pair(name, feat));
                    feats_.push_back(feat);
                }    
                fi.close();
                dimension_ = height_* width_;
            }


            void load_image_data(string path, int number_characters){
                for(int k = 0; k < number_characters; ++ k){
                    cv::Mat image = cv::imread(path+"/"+std::to_string(k)+".png");
                    if(image.empty()) continue;
                    cv::resize(image, image, cv::Size(width_,height_));
                    if(image.channels() == 3){
                        cvtColor(image, image, CV_RGB2GRAY);
                    }
                    std::shared_ptr<Element> feat(new Element());
                    feat->feat_.clear();
                    for(int i = 0; i < image.rows; ++ i){
                        for(int j = 0; j < image.cols; ++ j){
                            uchar v = image.at<uchar>(i,j);
                            feat->feat_.push_back(float(v)/ 255.0f);
                        }
                    }
                    feat->number_ = 1;
                    feat->label_name_ = std::to_string(k);
                    feat->index_ = k;
                    feats_.push_back(feat);
                }
                dimension_ = height_* width_;

            }



            void load_feat(string file){



            }


            // calculate covariance and mean using opencv
            float covariance(int i0, int i1){
                if( i1 <= i0 ) return 0.0f;
                cv::Mat temp = cv::Mat::zeros(dimension_, i1 -i0,  CV_32FC1);
                for(int i = i0; i< i1; ++ i){
                    for(int j = 0; j < dimension_; ++ j){
                       temp.at<float>(j, i-i0) = feats_[i]->feat_[j];
                       //LOG(INFO)<< feats_[i]->feat_[j];
                    }
                }
                cv::Mat cov;
                cv::Mat mean;
                cv::calcCovarMatrix(temp, cov, mean, CV_COVAR_NORMAL | CV_COVAR_COLS, CV_32F);
                LOG(INFO)<< cov.rows<<" "<<dimension_;
                //LOG(INFO)<< cov;
                float v = 0;
                for(int i = 0; i < dimension_; ++ i){
                    v += cov.at<float>(i,i);
                }
                //float v = cv::determinant(cov);
                //LOG(INFO)<< v;
                return v / float(dimension_);
            }

            float variance(int i0, int i1){
                if( i1 <= i0 ) return 0.0f;
                vector<float> mean(dimension_, 0.0f);
                for(int j = 0; j < dimension_; ++ j){
                    for(int i = i0; i < i1; ++ i){
                        mean[j] += feats_[i]->feat_[j];
                    }
                    mean[j] /= float(i1 -i0);
                }
                vector<float> var(dimension_, 0.0f);
                for(int j = 0; j < dimension_; ++ j){
                    for(int i = i0; i < i1; ++ i){
                        var[j] += (feats_[i]->feat_[j] - mean[j])*(feats_[i]->feat_[j] - mean[j]);
                    }
                    var[j] /= float(i1-i0);
                }
                float v = 0.0f;
                for(int j = 0; j < dimension_; ++ j){
                    v = std::max(var[j], v);
                }
                return v ;
            }

            cv::Mat mean(){
                cv::Mat img = cv::Mat::zeros(height_, width_, CV_8UC1);
                vector<float> mean(dimension_, 0.0f);
                for(int j = 0; j < dimension_; ++ j){
                    for(int i = 0; i < feats_.size(); ++ i){
                        mean[j] += feats_[i]->feat_[j];
                    }
                    mean[j] /= float(feats_.size());
                }
                for(int i = 0; i < height_; ++ i){
                    for(int j = 0; j < width_; ++ j){
                        img.at<uchar>(i,j) = 255* mean[i*width_+j];
                    }
                }
                return img;
            }


            void fast_variance(int i0, int i1, vector<float>& varl, vector<float>& varr ){
                varl.clear();
                varr.clear();
                //vector<vector<float> > s;
                //cv::Mat cv::Mat::zeros();
                 if( i1 <= i0 ) return;
                std::shared_ptr<float> xm(new float [(i1-i0)*dimension_](), std::default_delete<float []>());
                std::shared_ptr<float> x2(new float [(i1-i0)*dimension_](), std::default_delete<float []>());

                for(int i = i0; i < i1; ++i){
                    memcpy(xm.get() + (i-i0)*dimension_, feats_[i]->feat_.data(), sizeof(float)*dimension_);
                    memcpy(x2.get() + (i-i0)*dimension_, feats_[i]->feat_.data(), sizeof(float)*dimension_);
                    for(int j = 0; j < dimension_; ++ j){
                        x2.get()[(i-i0)*dimension_+j] *= x2.get()[(i-i0)*dimension_+j];
                    }
                }

                for(int j = 0; j < dimension_; ++ j){
                    for(int i = i0+1; i < i1; ++ i){
                        xm.get()[(i-i0)*dimension_+j] += xm.get()[(i-i0-1)*dimension_+j];
                        x2.get()[(i-i0)*dimension_+j] += x2.get()[(i-i0-1)*dimension_+j];
                    }
                }

                varl.resize(i1-i0);
                varr.resize(i1-i0);
                float ex2 = 0.0f, ex = 0.0f, v = 0.0f;
                varl[0] = 0.0f;
                varr[0] = 0.0f;
                for(int j = 0 ; j < dimension_; ++ j){
                    ex2 = x2.get()[(i1-1-i0)*dimension_+j] / float(i1-i0);
                    ex  = xm.get()[(i1-1-i0)*dimension_+j] / float(i1-i0);
                    v = ex2 -ex*ex;
                    varr[0] = std::max(v, varr[0]);
                }



                for(int i = i0+1; i < i1; ++ i){
                    varl[i-i0] = 0;
                    varr[i-i0] = 0;
                    for(int j = 0; j < dimension_; ++ j){
                        ex2 = x2.get()[(i-i0)*dimension_+j] / float(i-i0);
                        ex = xm.get()[(i-i0)*dimension_+j] / float(i-i0);
                        v = ex2 - ex*ex;
                        varl[i-i0] = std::max(v , varl[i-i0] );
                        ex2 = (x2.get()[(i1-1-i0)*dimension_+j] - x2.get()[(i-i0)*dimension_+j]) /float(i1 - i);
                        ex =  (xm.get()[(i1-1-i0)*dimension_+j] - xm.get()[(i-i0)*dimension_+j]) /float(i1 - i);
                        v = ex2 - ex*ex;
                        varr[i-i0] =std::max(v, varr[i-i0]);
                    }
                    

                }



            }





        public:
            vector< std::shared_ptr<Element> > feats_;

            int dimension_;
            int height_;
            int width_;
    };



}


#endif