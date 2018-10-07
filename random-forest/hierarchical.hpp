#ifndef CASCADE_HIERARCHICAL_HPP_
#define CASCADE_HIERARCHICAL_HPP_
#include <memory>
#include <map>
#include "random-forest/data.hpp"
#include "random-forest/node.hpp"
#include "random-forest/graphviz.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
namespace kurff{
    class Hierarchical{
        public:
            Hierarchical(int ncluster):node_index_(0), ncluster_(ncluster){

            }
            ~Hierarchical(){

            }

            void clustering(Data* data){
                std::shared_ptr<Node> p(new Node());
                p->node_index_ = node_index_;
                p->node_name_ = std::to_string(p->node_index_);
                tree_.insert(std::make_pair(node_index_, p));
                vector<int> data_index;
                for(int i = 0; i < data->feats_.size(); ++ i){
                    data_index.push_back(data->feats_[i]->index_);
                }
                clustering(p.get(), data, data_index);
            }

            void save_graphvis(string file){
                GraphViz<Node> graph;
                for(auto n : tree_){
                    //LOG(INFO)<<
                    graph.add_node(n.second.get());
                }

                for(auto n : tree_){
                    for(auto x : n.second->child_){
                        if(x != nullptr){
                            graph.add_edge(n.second.get(), x);
                        }
                    }
                }

                graph.render(file);

            }

            void clustering(Node* node, Data* data, const vector<int>& data_index){
                LOG(INFO)<<"data_index: "<< data_index.size();
                if( data_index.size() <=1 ){
                    //node->center_ = data->feats_[data_index[0]];
                    node->node_name_ = "node"+data->feats_[data_index[0]]->label_name_;
                    node->center_ = cv::Mat(data->height_, data->width_, CV_32FC1, data->feats_[data_index[0]]->feat_.data());
                    return;
                }
                int number = data_index.size();
                cv::Mat d = cv::Mat::zeros(number, data->dimension_, CV_32FC1);
                for(int i = 0; i < data_index.size(); ++ i){
                    memcpy(d.ptr<float>(i), data->feats_[data_index[i]]->feat_.data(), sizeof(float)*data->dimension_);
                }


                cv::kmeans(d, ncluster_, node->index_, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 50, 1.0),
                3,KMEANS_RANDOM_CENTERS, node->center_);

                std::map<int, std::shared_ptr<vector<int> > > data_index_new;
                for(int i = 0; i < ncluster_; ++ i){
                    std::shared_ptr<vector<int> > p(new vector<int>());
                    p->clear();
                    data_index_new.insert(std::make_pair(i, p));
                }

                for(int i = 0; i < number; ++ i){
                   int label =  node->index_.at<int>(i);
                   //LOG(INFO)<<"label: "<< label;
                   auto it = data_index_new.find(label);
                   if(it != data_index_new.end()){
                       it->second->push_back(data->feats_[data_index[i]]->index_);
                   }
                }

                node->child_.clear();
                for(int i = 0; i < ncluster_; ++ i){
                    std::shared_ptr<Node> p(new Node());
                    p->node_index_ = ++ node_index_;
                    p->node_name_ = std::to_string(p->node_index_);
                    tree_.insert(std::make_pair(node_index_, p));
                    node->child_.push_back(p.get());
                    auto it  = data_index_new.find(i);
                    if(it != data_index_new.end()){
                        clustering(p.get(), data, *(it->second) );
                    }
                }
            }
        protected:



        protected:
            map<int, std::shared_ptr<Node> > tree_;
            int node_index_;
            int ncluster_;




    };



}


#endif