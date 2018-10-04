#ifndef CASCADE_RANDOM_FOREST_NODE_HPP_
#define CASCADE_RANDOM_FOREST_NODE_HPP_
#include <vector>
#include <string>
#include <memory>
using namespace std;
namespace kurff{




    class Node{
        public:
            Node(){

            }
            ~Node(){

            }



        public:
            Node* left_;
            Node* right_;
            int node_index_;
            int feat_index_;
            float threshold_;
            string name_;



    };


}
#endif