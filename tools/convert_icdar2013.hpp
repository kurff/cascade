#ifndef CASCADE_TOOLS_CONVERT_ICDAR2013_HPP_
#define CASCADE_TOOLS_CONVERT_ICDAR2013_HPP_
#include <memory>
#include <string>
#include "opencv2/opencv.hpp"
#include "tools/convert.hpp"


namespace kurff{
    class ConvertICDAR2013 : public Convert{
        public:
            ConvertICDAR2013(const std::string& path, int resized_height, int resized_width) 
            : Convert(path, resized_height, resized_width){
                
            }

            ~ConvertICDAR2013(){

            }

            void read_train();

            void read_test();

        protected:
            


    };





} // end of namespace kurff



#endif