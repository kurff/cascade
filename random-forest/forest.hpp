#ifndef CASCADE_RANDOM_FOREST_FOREST_HPP_
#define CASCADE_RANDOM_FOREST_FOREST_HPP_
#include "random-forest/tree.hpp"
namespace kurff{
    class Forest{
        public:
            Forest(){

            }
            ~Forest(){

            }

            
        protected:
            std::map<string, std::shared_ptr<Tree> > forest_;



    };



}


#endif