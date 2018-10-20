#ifndef __XML_HPP__
#define __XML_HPP__
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/typeof/typeof.hpp>   
#include <boost/filesystem.hpp>
#include <string>
#include <vector>
#include <iostream>
#include "glog/logging.h"
using namespace std;

using boost::property_tree::ptree;
using boost::property_tree::xml_writer_settings;
namespace kurff{

    template<typename Object>
    class XML{
        public: 
            XML(){

            }
            ~XML(){

            }

            void read_voc_format(string file, vector<Object>& objects){
                objects.clear();
                ptree pt;
                read_xml(file.c_str(), pt);
                ptree child = pt.get_child("annotation"); 
                //cout<<child.get<string>("folder");
                ptree size = child.get_child("size");
                //cout<<size.get<int>("width")<<endl;
                //cout<<size.get<int>("height")<<endl;
                //cout<<size.get<int>("depth")<<endl;
                int height = size.get<int>("height");
                int width = size.get<int>("width");
               
                for(BOOST_AUTO(pos,child.begin());pos != child.end();++pos)  //boost中的auto  
                {  
                    if(pos->first=="object"){
                        //cout<<"\t"+pos->second.get<string>("name"); 
                        Object object;
                        ptree obj = pos->second.get_child("bndbox");
                        object.x = obj.get<float>("xmin");
                        object.y = obj.get<float>("ymin");
                        float x1 = obj.get<float>("xmax");
                        float y1 = obj.get<float>("ymax");
                        object.height = y1 - object.y;
                        object.width = x1 - object.x;
                        objects.push_back(object);
                    }
                }  
                //return objects;
            }

            void write_voc_format(string path,string file, const vector<Object>& object, int height, int width){
                ptree pt;
                ptree& node = pt.add("annotation","");
                node.add("folder","VOC2007");
                node.add("filename", file+".jpg");
                ptree& s = node.add("size","");
                s.add("width", width);
                s.add("height", height);
                s.add("depth", 3);
                
                for(auto obj : object){
                    //cout<<"write obj: "<< obj.name_<<endl;
                    //if(obj.confidence_ <= threshold) continue;
                    ptree& o = pt.add("annotation.object","");
                    o.add("name","character");
                    o.add("pose", "Unspecified");
                    o.add("truncated",0);
                    o.add("difficult",0);
                    //node = pt.add("annotation.object.bndbox","");
                    ptree& b = o.add("bndbox","");
                    b.add("xmin", obj.x);
                    b.add("ymin", obj.y);
                    b.add("xmax", obj.x + obj.width);
                    b.add("ymax", obj.y + obj.height);
                }
                
                write_xml(path+"/"+file+".xml", pt,std::locale(), boost::property_tree::xml_writer_make_settings<std::string>('\t', 1));
            }

            



        protected: 
            


    };






}






#endif