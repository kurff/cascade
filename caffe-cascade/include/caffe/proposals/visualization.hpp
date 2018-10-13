#ifndef __KURFF_VISUALIZATION_HPP__
#define __KURFF_VISUALIZATION_HPP__

#include<string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "caffe/proposals/box.hpp"


using namespace cv;
using namespace std;
namespace kurff{
void visualize(Mat& img, const vector<Box>& boxes, Scalar scalar);
}

#endif