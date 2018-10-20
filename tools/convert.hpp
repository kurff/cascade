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

#include "tools/data.hpp"
#include "tools/image_transform.hpp"
#include "tools/xml.hpp"



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
            Convert(const std::string& path, int resized_height, int resized_width):
            resized_height_(resized_height), resized_width_(resized_width){
                data_.reset(new Data());
                data_->path = path;
                image_transform_.reset(new ImageTransform(resized_height_,resized_width_));
                xml_.reset(new XML<cv::Rect>());
            }
            ~Convert(){

            }

            void init(const string& name , const string& backend);

            void read_splits();

            virtual void read_train() = 0;

            virtual void read_test() = 0;

           

            void write_train_data(int show = 0);

            void write_test_data(int show = 0);


            void write_db(const std::map<string, DataPair>& data, const std::string& name, int show);


        protected:
            std::shared_ptr<db::DB> db_;
            std::shared_ptr<db::Transaction> txn_;
            std::shared_ptr<Data> data_;
            std::shared_ptr<ImageTransform> image_transform_;
            int resized_height_;
            int resized_width_;
            std::shared_ptr<XML<cv::Rect> > xml_;
    };


}


#endif