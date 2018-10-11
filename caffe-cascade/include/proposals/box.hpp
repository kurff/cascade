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

        

        Box(const Box& box){
            x = box.x;
            y = box.y;
            height = box.height;
            width = box.width;
            confidence_ = box.confidence_;
            name_ = box.name_;
            predict_ = box.predict_;
        }

        bool check(int rheight, int rwidth){
            if(x < 0 || y <0 || width < 0 || height < 0 ){
                return false;
            }

            x = std::max(0,x);
            y = std::max(0,y);

            int x1 = std::min( x + width, rwidth-1);
            int y1 = std::min( y + height, rheight-1);

            width = x1 - x;
            height = y1 - y; 
            //LOG(INFO)<<x<<" "<<y<<" "<<width<<" "<<height;
            if(width <= 1 || height <= 1 ){
                return false;
            }else{
                return true;
            }

        }

    public:
        float confidence_;
        string name_;
        int predict_;
        

};


bool compare(const Box & b0, const Box& b1){
    return b0.confidence_ < b1.confidence_;
}












}







#endif