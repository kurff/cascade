#ifndef CASCADE_RANDOM_FOREST_DATA_HPP_
#define CASCADE_RANDOM_FOREST_DATA_HPP_
#include <vector>
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
namespace kurff{
    class Data{
        public:
            Data(){


            }

            ~Data(){


            }

            void compute_feature(){

            }
            void load_data(string file){
                cv::Mat image = cv::imread(file);
                

            }

            void load_feat(){


            }

        protected:
            vector<vector<float> > feat_;
            vector<int> num_examples_;
            std::map<string , int > name2index_;
            std::map<int,  string> index2name_;



    };



}


#endif