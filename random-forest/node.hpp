#ifndef CASCADE_RANDOM_FOREST_NODE_HPP_
#define CASCADE_RANDOM_FOREST_NODE_HPP_
#include <vector>
#include <string>
#include <memory>
using namespace std;

#include "random-forest/classifier.hpp"
namespace kurff{




    class Node{
        public:
            Node(){

            }
            Node(){

            }



        public:
            Node* left_;
            Node* right_;
            std::shared_ptr<Classifier> classifier_;
            int index_;
            



    };


}
#endif