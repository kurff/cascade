#ifndef __KURFF_BOX_HPP__
#define __KURFF_BOX_HPP__

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
using namespace std;
namespace kurff{


class Box: public cv::Rect{
    public:
        Box():cv::Rect(0,0,0,0){
        }
        ~Box(){

        }
        Box(const Box& box);
        bool check(int rheight, int rwidth);
    public:
        float confidence_;
        string name_;
        int predict_;
};


bool compare(const Box & b0, const Box& b1);

}







#endif