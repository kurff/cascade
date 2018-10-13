#ifndef __CASCADE_HPP__
#define __CASCADE_HPP__
#include <string>
#include <vector>
#include "core/box.hpp"
#include "core/registry.h"
#include "model/regression.hpp"
#include "opencv2/opencv.hpp"


#include <iostream>
#include <fstream>
#include <vector>
using namespace cv;
using namespace std;
namespace kurff{
    class Cascade{
        public:
            Cascade(){

            }
            ~Cascade(){

            }

            virtual bool init(string file) = 0;

            virtual void run(const Mat& image, const Box& box, Box& box_next) = 0;

            virtual void run(const Mat& image, const vector<Box>& boxes, vector<Box>& boxes_next) = 0;
        
        protected:





    };


    CAFFE_DECLARE_REGISTRY(CascadeRegistry, Cascade);
    CAFFE_DEFINE_REGISTRY(CascadeRegistry, Cascade);

    class CascadeRegression: public Cascade{
        public:
            CascadeRegression(){

            }
            ~CascadeRegression(){

            }

            // initialization

            bool init(string file){




            }


            void run(const Mat& image, const Box& box, Box& box_next){
                int x0 = box.x;
                int y0 = box.y;
                int x1 = box.x + box.width;
                int y1 = box.y + box.height;
                assert(theta.size() ==4);

                


            }

            void run(const Mat& image, const vector<Box>& boxes, vector<Box>& boxes_next){
                boxes_next.clear();
                for(auto b : boxes){
                    Box box_next;
                    run(image, b, box_next);
                    boxes_next.push_back(box_next);
                }
            }


        protected:
            std::vector<std::shared_ptr<Model> > cascade_regression_;
            

    };
    CAFFE_REGISTER_CLASS(CascadeRegistry, CascadeRegression, CascadeRegression);


    class DynamicCascadeRegression: public Cascade{
        public:
            DynamicCascadeRegression(){

            }
            ~DynamicCascadeRegression(){

            }

            // initialization

            bool init(string file){




            }


            void run(const Mat& image, const Box& box, Box& box_next){
                int x0 = box.x;
                int y0 = box.y;
                int x1 = box.x + box.width;
                int y1 = box.y + box.height;
                assert(theta.size() ==4);

                


            }

            void run(const Mat& image, const vector<Box>& boxes, vector<Box>& boxes_next){
                boxes_next.clear();
                for(auto b : boxes){
                    Box box_next;
                    run(image, b, box_next);
                    boxes_next.push_back(box_next);
                }
            }


        protected:
            std::vector<std::shared_ptr<Model> > cascade_regression_;
                        

    };
    CAFFE_REGISTER_CLASS(CascadeRegistry, DynamicCascadeRegression, DynamicCascadeRegression);

} // end of namespace kurff

#endif