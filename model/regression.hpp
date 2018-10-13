#ifndef __REGRESSION_HPP__
#define __REGRESSION_HPP__
#include "model/model.hpp"
namespace kurff{
    class Regression : public Model{
        public:

            Regression(int top_k) : Model(top_k){

            }
            ~Regression(){

            }

            void run(const Mat& image, vector<Box>& objects){

            }

            void run_each(const Mat& image, vector<float>& confidence, vector<int>& label){


            }
        
        protected:
            

    };
    CAFFE_REGISTER_CLASS(ModelRegistry, Regression, Regression);
}



#endif