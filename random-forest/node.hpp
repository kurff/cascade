#ifndef CASCADE_RANDOM_FOREST_NODE_HPP_
#define CASCADE_RANDOM_FOREST_NODE_HPP_
#include <vector>
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
namespace kurff{




    class Node{
        public:
            Node(){

            }
            ~Node(){

            }
            string name(){
                return node_name_;
            }


        public:
            Node* left_;
            Node* right_;
            vector<Node*> child_;
            int node_index_;
            int feat_index_;
            float threshold_;
            string name_;
            string node_name_;
            bool is_split_;
            cv::Mat center_;
            cv::Mat index_;



    };


}
#endif