#ifndef __GRAPHVIZ_HPP__
#define __GRAPHVIZ_HPP__
#include "gvc.h"
#include <map>
#include "glog/logging.h"
using namespace std;
namespace kurff{
    template<typename T>
    class GraphViz{
        
        public:
            GraphViz(){
                gvc_ = gvContext();
                //string out="-o"+file;
                //const char* argv[4] = {"exe","-Kdot","-Tpng","-ograph.png"};
                //gvParseArgs(gvc_, 4 , const_cast<char* *>(argv));
                //string name = "graph";
                g_ = agopen(const_cast<char*>("tmp"), Agdirected, 0);
            }
            ~GraphViz(){
                gvFreeLayout(gvc_, g_);
                agclose(g_);
                gvFreeContext(gvc_);
            }

            Agnode_t* add_node(T* node){
                LOG(INFO)<< "creating "<< node->name();
                Agnode_t* t = agnode(g_, const_cast<char*>(node->name().c_str()), true);
                nodes_.insert(std::make_pair(node->name(), t));
                return t;
            }

            // Agnode_t* add_node(T* node, char* name, char* value){
            //     Agnode_t* n = add_node(node);
            //     //agsafeset(n, "color", "red");
            // }





            Agedge_t* add_edge(T* parent, T* child){
                auto prt = nodes_.find(parent->name());
                auto cid = nodes_.find(child->name());
                if(prt == nodes_.end()){
                    LOG(INFO)<<"can not find node "<< parent->name();
                    return nullptr;
                }
                if(cid == nodes_.end()){
                    LOG(INFO)<<"can not find node" << child->name();
                    return nullptr;
                }
                string name = parent->name()+"_"+child->name();
                Agedge_t* edge = agedge(g_,prt->second, cid->second, const_cast<char*>(name.c_str()), 1);
                edges_.insert(std::make_pair(name.c_str(), edge ));
                return edge;
            }



            void render(string file){
                //gvLayoutJobs(gvc_, g_);
                
                gvLayout(gvc_,g_, "dot");
                //attach_attrs(g_);
                /* Write the graph according to -T and -o options */
                //gvRenderJobs(gvc_, g_);
                gvRenderFilename(gvc_,g_,"png", file.c_str());
                //attach_attrs(g_);
            }

        protected:
            Agraph_t *g_;
            std::map<string, Agnode_t*> nodes_;
            std::map<string, Agedge_t*> edges_;
            GVC_t *gvc_;

    };




}









#endif