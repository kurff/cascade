
#include "caffe/proposals/utils.hpp"
namespace kurff{

float overlap(const Box& b0, const Box& b1){
    int b0_x0 = b0.x;
    int b0_y0 = b0.y;
    int b0_x1 = b0.x + b0.width;
    int b0_y1 = b0.y + b0.height;

    int b1_x0 = b1.x;
    int b1_y0 = b1.y;
    int b1_x1 = b1.x + b1.width;
    int b1_y1 = b1.y + b1.height;

    if(b0_y0 > b1_y1){
        return 0.0f;
    }
    if(b0_x0 > b1_x1){
        return 0.0f;
    }
    if(b0_y1 < b1_y0){
        return 0.0f;
    }
    if(b0_x1 < b1_x0){
        return 0.0f;
    }

    float x = min(b0_x1,b1_x1) - max(b0_x0,b1_x0);
    float y = min(b0_y1,b1_y1) - max(b0_y0,b1_y0);
    float intersection = x*y;
    float a0 = (b0_y1-b0_y0)*(b0_x1-b0_x0);
    float a1 = (b1_y1-b1_y0)*(b1_x1-b1_x0);
    return intersection /(a0+a1-intersection);
}

Box expand_box(const Box& box, float ratio, int height, int width){
    float cx = float(box.x) + float(box.width)/2.0f;
    float cy = float(box.y) + float(box.height)/2.0f;

    Box box_new;

    float width_box = float(box.width)/2.0f * ratio;
    float height_box = float(box.height)/2.0f * ratio;
    box_new.x = (int)std::max(0.0f, cx - (width_box));
    box_new.y = (int)std::max(0.0f, cy - (height_box));
    
    float ex = std::min(float(width-1), cx + (width_box));
    float ey = std::min(float(height-1), cy + (height_box));

    box_new.width = int(ex - box_new.x);
    box_new.height = int(ey - box_new.y);
    return box_new;
}


}