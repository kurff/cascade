#include "tools/convert.hpp"

namespace kurff{
    void Convert::init(const string& name, const string& backend){
        db_.reset(db::GetDB(backend));
        db_->Open(name, db::NEW);
        txn_.reset(db_->NewTransaction());

    }

    void Convert::read_splits(){
        read_train();
        read_test();
    }

    void Convert::write_db(const std::map<string, DataPair>& data, const std::string& name, int show){
        
        db_.reset(db::GetDB("lmdb"));
        db_->Open(name.c_str(), db::NEW);
        txn_.reset(db_->NewTransaction());
        ImageData image_data;
        for(auto d : data){
            image_data.image = cv::imread(d.second.image_name);
            xml_->read_voc_format(d.second.annotation_name,image_data.objects);
            image_transform_->forward(image_data);
            if(show){              
                for(auto r : image_data.objects){
                    cv::rectangle(image_data.image, r, Scalar(0,0,255),2);
                }
                cv::imshow("src", image_data.image);
                cv::waitKey(0);
            }

            


        }



        



    }

    void Convert::write_train_data(int show){
        write_db(data_->train, data_->path+"/train_lmdb", show);
    }

    void Convert::write_test_data(int show){
        write_db(data_->test, data_->path+"/test_lmdb", show);
    }
}
