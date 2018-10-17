#include "tools/image_transform.hpp"
using namespace kurff;
int main(int argc, char* argv[]){
    ImageTransform* trans = new ImageTransform(600, 1200);
    for(int i = 100; i < 300; ++ i){
        cv::Mat image = cv::imread(argv[1]+std::to_string(i)+".jpg");
        cv::Mat output;
        trans->forward(image, output);
        cv::imshow("src",output);
        cv::waitKey(0);
    }


    return 0;
}