#ifndef CASCADE_RANDOM_FOREST_TREE_HPP_
#define CASCADE_RANDOM_FOREST_TREE_HPP_

#include <map>
#include <algorithm>
#include "glog/logging.h"
#include "random-forest/node.hpp"
#include "random-forest/data.hpp"
#include "random-forest/random.hpp" 
#include "glog/logging.h"
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
                    node->node_index_ = index;
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


            
            void train(Data* data){
                nodes_index_ = 0;
                std::shared_ptr<Node> p(new Node());
                p->node_index_ = nodes_index_;
                nodes_.insert(std::make_pair(nodes_index_, p));
                train(p.get(), data, 0, data->feats_.size(), 0);
            }

            void save(string file){
                

            }

            void train(Node* node, Data* data, int i0, int i1, int depth){
                indexes_.clear();
                for(int i = 0; i < num_feature_stage_; ++ i){
                    int index = random_->Next(0, data->dimension_);
                    indexes_.push_back(index);
                }
                LOG(INFO)<<"train node: "<< node->node_index_<<" "<<i0<<" "<<i1;

                float max_vars = -1e10;
                float vars = 0.0f;
                float v0 = 0.0f, v1 = 0.0f;
                float threshold = 0.0f;

                int feat_index = 0;
                int data_index = 0;

                for(int idx = 0; idx < num_feature_stage_; ++ idx){
                    for(int j = i0; j < i1 ; ++ j ){
                        data->feats_[j]->v_ = data->feats_[j]->feat_[indexes_[idx]];

                    }
                    LOG(INFO)<<"sorting: "<< data->feats_.size();
                    std::sort(data->feats_.begin() + i0, data->feats_.begin() + i1, kurff::compare );
                    LOG(INFO)<<"finish sorting: ";
                    float v = data->variance(i0, i1);
                    
                    //LOG(INFO)<<"covariance: "<< v;
                    for(int i = i0; i < i1; ++ i){
                        v0 = data->variance(i0,i);
                        v1 = data->variance(i,i1);
                        vars = v - v0*float(i-i0)/ float(i1 -i0) - v1*float(i1 - i) /float(i1 - i0);
                        //LOG(INFO)<<"v: "<<v<<" v0: "<< v0<< " v1: "<< v1 <<" vars: "<< vars;
                        if(max_vars <= vars){
                            max_vars = vars;
                            feat_index = indexes_[idx];
                            threshold = data->feats_[i]->v_;
                            data_index = i;
                            LOG(INFO)<<"max vars: "<< max_vars<<" feat_index: "<< feat_index<<" threshold: "
                            <<threshold<<" data_index: "<< data_index <<" i0: "<< i0 <<" i1: "<< i1 ;
                        }
                    }
                }
                LOG(INFO)<<"Best: "<< max_vars<<" feat_index: "<< feat_index<<" threshold: "
                <<threshold<<" data_index: "<< data_index <<" i0: "<< i0 <<" i1: "<< i1 ;
                node->feat_index_ = feat_index;
                node->threshold_ = threshold;
                //nodes_.insert(std::make_pair(nodes_index_, p));
                if(depth <= max_depth_ && i1 - i0 > 1 ){
                    std::shared_ptr<Node> l (new Node());
                    node->left_ = l.get();
                    ++ nodes_index_;
                    l->node_index_ = nodes_index_;
                    nodes_.insert(std::make_pair(nodes_index_, l));
                    train(l.get(), data, i0, data_index, depth + 1);
                    
                    std::shared_ptr<Node> r(new Node());
                    node->right_ = r.get();
                    ++ nodes_index_;
                    r->node_index_ = nodes_index_;
                    nodes_.insert(std::make_pair(nodes_index_, r));
                    train(r.get(), data, data_index, i1, depth + 1);
                }else{
                    node->left_ = nullptr;
                    node->right_ = nullptr;
                    node->name_ = data->feats_[i0]->label_name_;
                }

            }
        protected:
            std::map<int, std::shared_ptr<Node> > nodes_;
            std::shared_ptr<Random> random_;
            std::vector<int> indexes_;
            int num_feature_stage_;
            int max_depth_;
            int nodes_index_;
    };
}


#endif