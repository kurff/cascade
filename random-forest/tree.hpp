#ifndef CASCADE_RANDOM_FOREST_TREE_HPP_
#define CASCADE_RANDOM_FOREST_TREE_HPP_

#include <map>
#include "glog/logging.h"
#include "random-forest/node.hpp"
#include "random-forest/data.hpp"
#include "random-forest/random.hpp" 
using namespace std;

namespace kurff{
    class Tree{
        public:
            Tree(int num_feature_stage, int max_depth) : num_feature_stage_(num_feature_stage),
            max_depth_(max_depth){
                random_.reset(new Random());

            }
            ~Tree(){

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


            

            void train(const Data& data, int depth){
                if(depth >= max_depth_){

                }
                indexes_.clear();
                for(int i = 0; i < num_feature_stage_; ++ i){
                    int index = random_->Next(0, data.dimension_);
                    indexes_.push_back(index);
                }

                

                






            }



            

        protected:
            std::map<int, std::shared_ptr<Node> > nodes_;
            std::shared_ptr<Random> random_;
            std::vector<int> indexes_;
            int num_feature_stage_;
            int max_depth_;





    };


}


#endif