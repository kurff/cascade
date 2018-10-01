#ifndef CASCADE_TOOLS_CONVERT_ICDAR2013_HPP_
#define CASCADE_TOOLS_CONVERT_ICDAR2013_HPP_

#include "tools/convert.hpp"

namespace kurff{
    class ConvertICDAR2013 : public Convert{
        public:
            ConvertICDAR2013() : Convert(){

            }

            ~ConvertICDAR2013(){

            }

            void read_data_label();

            void write_db();


        protected:



    };





} // end of namespace kurff



#endif