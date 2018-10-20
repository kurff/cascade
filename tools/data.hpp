#ifndef TOOLS_DATA_HPP_
#define TOOLS_DATA_HPP_
#include "opencv2/opencv.hpp"

    typedef struct ImageData_{
        cv::Mat image;
        std::vector<cv::Rect> objects;
    }ImageData;


#endif