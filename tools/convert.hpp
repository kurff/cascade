#ifndef CASCADE_TOOLS_CONVERT_HPP_
#define CASCADE_TOOLS_CONVERT_HPP_
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "tools/image_transform.hpp"




using namespace cv;
using namespace std;
using namespace caffe;
namespace kurff{

    typedef struct DataPair_{
        std::string image_name;
        std::string annotation_name;
    }DataPair;

    typedef struct Data_{
        std::map<string, DataPair> train;
        std::map<string, DataPair> test;
        std::string path;
    }Data;


    class Convert{
        public:
            Convert(const std::string& path){
                data_.reset(new Data());
                data_->path = path;

            }
            ~Convert(){

            }

            void init(const string& name , const string& backend);

            void read_splits();

            virtual void read_train() = 0;

            virtual void read_test() = 0;

            virtual void read_train_data(int show) = 0;

            virtual void read_test_data(int show) = 0;

            void write_db();


        protected:
            std::shared_ptr<db::DB> db_;
            std::shared_ptr<db::Transaction> txn_;
            std::shared_ptr<Data> data_;


    };


}


#endif