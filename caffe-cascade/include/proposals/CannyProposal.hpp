#ifndef __CANNY_PROPOSAL__
#define __CANNY_PROPOSAL__
#include "proposals/Proposal.hpp"
#include "opencv2/opencv.hpp"
#include "utils/utils.hpp"
#include <memory>
namespace kurff{
    class CannyProposal : public Proposal{
        public:
            CannyProposal(int ratio = 3, int lowThreshold = 20, int kernel_size = 3): Proposal()
            , ratio_(ratio), lowThreshold_(lowThreshold), kernel_size_(kernel_size), ratio_size_(1.2f){
                this->name_="Canny";
            }
            ~CannyProposal(){

            }

            void run(const Mat& gray, vector<Box>& proposals){
                cv::Mat gray;
                cv::Mat edges;
                //LOG(INFO)<< "row: "<< image.rows;               
                Canny( gray, edges, 20, 100, kernel_size_ );
                cv::Mat labels, stats, centroids;
                int number = cv::connectedComponentsWithStats (edges, labels, stats, centroids);
                //for()
                //LOG(INFO)<< "rows: " << stats.rows <<" cols: "<<stats.cols;
                //imshow("canny", edges);
                //waitKey(0);
                proposals.clear();
                for(int i = 1; i < stats.rows; ++ i){
                    Box box;
                    box.height = stats.at<int>(i, CC_STAT_HEIGHT);
                    box.width = stats.at<int>(i, CC_STAT_WIDTH );                   
                    box.x = stats.at<int>(i,CC_STAT_LEFT ) ;
                    box.y = stats.at<int>(i,CC_STAT_TOP);
                    // box too small
                    if(box.height <5 || box.width < 5){
                        continue;
                    }
                    Box box_new = expand_box(box, 1.2, gray.rows, gray.cols);
                    proposals.push_back(box_new);
                }
            }
        protected:
            int ratio_;
            int lowThreshold_;
            int kernel_size_;
            float ratio_size_;
            


    };
    CAFFE_REGISTER_CLASS(ProposalRegistry, CannyProposal, CannyProposal);

}




#endif