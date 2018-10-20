#ifndef CASCADE_TOOLS_CONVERT_ICDAR2013_HPP_
#define CASCADE_TOOLS_CONVERT_ICDAR2013_HPP_
#include <memory>
#include <string>
#include "opencv2/opencv.hpp"
#include "tools/convert.hpp"
#include "tools/xml.hpp"

namespace kurff{
    class ConvertICDAR2013 : public Convert{
        public:
            ConvertICDAR2013(const std::string& path) : Convert(path){
                xml_.reset(new XML<cv::Rect>());
            }

            ~ConvertICDAR2013(){

            }

            void read_train();

            void read_test();

            void read_train_data(int show);

            void read_test_data(int show);




        protected:
            std::shared_ptr<XML<cv::Rect> > xml_;


    };





} // end of namespace kurff



#endif