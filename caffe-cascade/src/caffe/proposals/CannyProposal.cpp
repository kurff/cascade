#include "caffe/proposals/CannyProposal.hpp"
namespace kurff{


void CannyProposal::run(const Mat& gray, vector<Box>& proposals){
    cv::Mat edges;
    Canny( gray, edges, 20, 100, kernel_size_ );
    //LOG(INFO)<<"value: "<<gray.at<uchar>(0,0);
    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats (edges, labels, stats, centroids);
    proposals.clear();
    for(int i = 1; i < stats.rows; ++ i){
        Box box;
        box.height = stats.at<int>(i, CC_STAT_HEIGHT);
        box.width = stats.at<int>(i, CC_STAT_WIDTH );                   
        box.x = stats.at<int>(i,CC_STAT_LEFT ) ;
        box.y = stats.at<int>(i,CC_STAT_TOP);
        if(box.height * box.width < this->min_size_*this->min_size_){
            continue;
        }
        Box box_new = expand_box(box, 1.2, gray.rows, gray.cols);
        proposals.push_back(box_new);
    }
    #ifdef DEBUG
        cv::Mat vis;
        gray.copyTo(vis);
        visualize(vis, proposals, cv::Scalar(0,0,255));
        cv::imshow("src", vis);
        cv::imshow("edge", edges);
        cv::waitKey(0);
    #endif
}
CAFFE_REGISTER_CLASS(ProposalRegistry, CannyProposal, CannyProposal);





}