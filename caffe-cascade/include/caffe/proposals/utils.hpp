#ifndef __KURFF_UTILS_HPP_
#define __KURFF_UTILS_HPP_
#include <vector>
#include <cmath>
#include "glog/logging.h"
#include "caffe/proposals/box.hpp"

using namespace std;
namespace kurff{
float overlap(const Box& b0, const Box& b1);
Box expand_box(const Box& box, float ratio, int height, int width);

}







#endif

