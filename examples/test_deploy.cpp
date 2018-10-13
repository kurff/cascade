#include "model/model.hpp"
#include "model/classifier.hpp"

using namespace kurff;
int main(){
    std::shared_ptr<Model> ptr = ModelRegistry()->Create("Classifier", 5);
    ptr->init("../examples/test_deploy.prototxt", "",1);
    vector<float> confidence;
    vector<int> label;
    cv::Mat image = cv::imread("../examples/100.jpg");
    ptr->run_each(image,confidence,label);


    return 0;
}