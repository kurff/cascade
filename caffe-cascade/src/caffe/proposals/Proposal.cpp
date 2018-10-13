
#include "caffe/proposals/Proposal.hpp"

namespace kurff{
    string Proposal::name(){return name_;}
    template<typename Dtype>
    void Proposal::run(const Dtype* data, int height, int width, int channel, vector<Box>& proposals){
        cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
        for(int i = 0; i < height; ++ i){
            for(int j = 0; j < width; ++ j){
                uchar b = uchar(data[i*3*width + j *3]);
                uchar g = uchar(data[i*3*width + j*3 + 1]);
                uchar r = uchar(data[i*3*width + j*3 + 2]);
                        //LOG(INFO)<<data[i*3*width + j *3];
                        //std::cout<<"r: "<<int(r) <<"g: "<<int(g) <<"b: "<<int(b);
                image.at<Vec3b>(i,j) = Vec3b(b,g,r);

            }
        }
        cv::Mat gray;
        cvtColor( image, gray, CV_BGR2GRAY ); 
        run(gray, proposals);
        evaluate(gray, proposals);
    }
    void Proposal::evaluate(const Mat& image, vector<Box>& proposals){
        for(int i = 0; i < proposals.size(); ++ i){
            Mat sub = image(proposals[i]);
            Mat dst;
            equalizeHist(sub,dst);
            Mat m, d;
            meanStdDev(sub, m, d);
            proposals[i].confidence_ = d.at<double>(0,0);
        }
    }
 

    CAFFE_DEFINE_REGISTRY(ProposalRegistry, Proposal);
    
    template void Proposal::run<float>(const float*, int, int, int, vector<Box>&);
    template void Proposal::run<double>(const double*, int , int, int, vector<Box>&);
}