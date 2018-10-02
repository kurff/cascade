#ifndef CASCADE_RANDOM_FOREST_DATA_HPP_
#define CASCADE_RANDOM_FOREST_DATA_HPP_
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <assert.h>
using namespace cv;
using namespace std;
namespace kurff{
    typedef struct Element_{
        vector<float> feat_;
        int number_;
    }Element;



    class Data{
        public:
            Data(){


            }

            ~Data(){


            }

            void load_image_data(string file){
                ifstream fi(file.c_str(), ios::in);
                string name;
                string image_name;
                int number;
                int cnt = 0;
                while(fi >> name >> image_name >> number){
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
                    
                    feat->number_ = number;
                    //feat_.insert(std::make_pair(name, feat));
                    feat_.push_back(feat);
                }    
                fi.close();
                dimension_ = height_* width_;
            }

            void load_feat(string file){



            }


            float covariance(int i1, int i2){
                vector<float> mean(dimension_, 0.0f);
                for(int i = 0; i < feat_.size(); ++ i){
                    
                    for(int j = 0; j < dimension_; ++ j){

                    }
                }




            }






        public:
            vector< std::shared_ptr<Element> > feat_;
            int dimension_;
            int height_;
            int width_;
            

            

    };



}


#endif