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
        AnnoDatum anno_datum;
        Datum* datum = anno_datum.mutable_datum();
        int id = 0;
        for(auto d : data){
            image_data.image = cv::imread(d.second.image_name);
            xml_->read_voc_format(d.second.annotation_name,image_data.objects);
            image_transform_->forward(image_data);
            anno_datum.set_item_id(++id);
            if(show){              
                for(auto r : image_data.objects){
                    cv::rectangle(image_data.image, r, Scalar(0,0,255),2);
                }
                cv::imshow("src", image_data.image);
                cv::waitKey(0);
            }

            CVMatToDatum( image_data.image, datum);
            datum->set_label(1);
            anno_datum.clear_bbox();
            for(auto r : image_data.objects){
                BBoxProto* bbox = anno_datum.add_bbox();
                bbox->set_xmin(float(r.x));
                bbox->set_ymin(float(r.y));
                bbox->set_xmax(float(r.x+r.width));
                bbox->set_ymax(float(r.y+r.height));
                bbox->set_label(1);
            }

            string key_str = d.first;
            string out;
            CHECK(anno_datum.SerializeToString(&out));
            txn_->Put(key_str, out);
            if(id % 100 ==0){
                txn_->Commit();
                txn_.reset(db_->NewTransaction());
                LOG(INFO)<<"Processed "<<id<<" files";
            }
        }

        if(id % 100 !=0){
            txn_->Commit();
            LOG(INFO)<<"Processed "<<id<<" files";
        }
    }

    void Convert::write_train_data(int show){
        write_db(data_->train, data_->path+"/train_lmdb", show);
    }

    void Convert::write_test_data(int show){
        write_db(data_->test, data_->path+"/test_lmdb", show);
    }
}
