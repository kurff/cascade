#ifndef CASCADE_RANDOM_FOREST_GRAPH_HPP_
#define CASCADE_RANDOM_FOREST_GRAPH_HPP_

#include <map>
#include "glog/logging.h"
#include "random-forest/node.hpp"
using namespace std;

namespace kurff{
    class Graph{
        public:
            Graph(){

            }
            ~Graph(){

            }

            void insert_node(int index){
                auto it = nodes_.find(index);
                if(it == nodes_.end()){
                    std::shared_ptr<Node> node (new Node());
                    node->index_ = index;
                    node->left_ = nullptr;
                    node->right_ = nullptr;
                    nodes_.insert(std::make_pair(index, node));
                }
                else{
                    LOG(INFO)<< "graph have "<< index <<" node!";
                }
            }

            void insert_left(int index, int left){
                auto it = nodes_.find(index);
                auto itl = nodes_.find(left);
                if(it != nodes_.end() && itl != nodes_.end()){
                    it->second->left_ = itl->second.get(); 
                }
            }

            void insert_right(int index, int right){
                auto it = nodes_.find(index);
                auto itr = nodes_.find(right);
                if(it != nodes_.end() && itr != nodes_.end()){
                    it->second->right_ = itr->second.get(); 
                }
            }

            

        protected:
            std::map<int, std::shared_ptr<Node> > nodes_;





    };


}


#endif