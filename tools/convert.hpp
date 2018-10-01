#ifndef CASCADE_TOOLS_CONVERT_HPP_
#define CASCADE_TOOLS_CONVERT_HPP_
#include <string>
#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"


using namespace cv;
using namespace std;
using namespace caffe;
namespace kurff{
    class Convert{
        public:
            Convert(){


            }
            ~Convert(){

            }

            void init(const string& name , const string& backend);

            virtual void read_data_label() = 0;

            virtual void write_db() = 0;



        protected:
            std::shared_ptr<db::DB> db_;
            std::shared_ptr<db::Transaction> txn_;
            cv::Mat image_;
            vector<float> label_;



    };


}


#endif