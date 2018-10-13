
#include "caffe/proposals/Proposal.hpp"

namespace kurff{
    string Proposal::name(){return name_;}
    template<typename Dtype>
    void Proposal::run(const Dtype* data, int height, int width, int channel, vector<Box>& proposals){
        cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
        for(int i = 0; i < height; ++ i){
            for(int j = 0; j < width; ++ j){
                uchar b = uchar(data[i*width + j]);
                uchar g = uchar(data[height*width+i*width + j]);
                uchar r = uchar(data[2*height*width+i*width + j]);
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
 

    CAFFE_DEFINE_REGISTRY(ProposalRegistry, Proposal, int);
    
    template void Proposal::run<float>(const float*, int, int, int, vector<Box>&);
    template void Proposal::run<double>(const double*, int , int, int, vector<Box>&);
}