#ifndef CASCADE_RNADOM_FOREST_CLASSIFIER_HPP_
#define CASCADE_RNADOM_FOREST_CLASSIFIER_HPP_
namespace kurff{
    class Classifier{
        public:
            Classifier(){

            }
            ~Classifier(){

            }

            bool eval(float v, float t){
                if( v >= t){
                    return false;
                }
                else{
                    return true;
                }
            }

            void set(float t){
                t_ = t;
            }
        protected:
            float t_;


    };



}


#endif