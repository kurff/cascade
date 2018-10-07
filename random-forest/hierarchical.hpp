#ifndef CASCADE_HIERARCHICAL_HPP_
#define CASCADE_HIERARCHICAL_HPP_
#include <memory>
#include <map>
#include <fstream>
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
                
                p->center_ = data->mean();
                //cv::imshow("mean", p->center_);
                //cv::waitKey(0);
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
                    cv::imwrite("node/"+std::to_string(n.second->node_index_)+".png", n.second->center_);
                    for(int i = 0; i < n.second->codes_.size(); ++ i){
                        std::cout<< n.second->codes_[i]<<" ";
                    }
                    std::cout<<std::endl;
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

            void save_codes(string file){
                ofstream fo(file.c_str(), ios::out);
                for(auto n : tree_){
                    if(n.second->child_.size() ==0){
                        fo << n.second->data_index_<<" ";
                        for(auto c : n.second->codes_){
                            fo << c <<" ";
                        }
                        fo << std::endl;
                    }
                }
                fo.close();

            }

            void clustering(Node* node, Data* data, const vector<int>& data_index){
                LOG(INFO)<<"clustering: "<< data_index.size();
                if( data_index.size() <=1 ){
                    //node->center_ = data->feats_[data_index[0]];
                    node->node_name_ = "node"+data->feats_[data_index[0]]->label_name_;
                    node->center_ = cv::Mat::zeros(data->height_, data->width_, CV_8UC1);
                    node->data_index_ = data_index[0];
                    for(int i = 0; i < data->height_; ++i){
                        for(int j = 0; j < data->width_; ++ j){
                            node->center_.at<uchar>(i,j) = data->feats_[data_index[0]]->feat_[i*data->width_+j]*255;
                        }
                    }
                    //node->center_ = cv::Mat(data->height_, data->width_, CV_32FC1, data->feats_[data_index[0]]->feat_.data());
                    return;
                }
                int number = data_index.size();
                cv::Mat d = cv::Mat::zeros(number, data->dimension_, CV_32FC1);
                for(int i = 0; i < data_index.size(); ++ i){
                    memcpy(d.ptr<float>(i), data->feats_[data_index[i]]->feat_.data(), sizeof(float)*data->dimension_);
                }


                cv::Mat center;
                cv::kmeans(d, ncluster_, node->index_, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 50, 1.0),
                3,KMEANS_PP_CENTERS, center);



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
                    p->codes_.insert(p->codes_.end(), node->codes_.begin(), node->codes_.end());
                    p->codes_.push_back(i);
                    p->node_index_ = ++ node_index_;
                    p->node_name_ = std::to_string(p->node_index_);
                    p->center_ = cv::Mat::zeros(data->height_, data->width_, CV_8UC1);
                    //LOG(INFO)<< center.rows<<" "<< center.cols;
                    for(int j = 0; j < data->height_; ++ j){
                        for(int q = 0; q < data->width_; ++ q){
                            p->center_.at<uchar>(j,q) = center.at<float>(i, j*data->width_ + q)*255;
                        }
                    }

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