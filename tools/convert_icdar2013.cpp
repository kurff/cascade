#include "tools/convert_icdar2013.hpp"
#include <fstream>
using namespace std;
namespace kurff{
    void ConvertICDAR2013::read_train_data(int show){
        for(auto d : data_->train){
            cv::Mat image = cv::imread(d.second.image_name);
            vector<Rect> objects;
            xml_->read_voc_format(d.second.annotation_name,objects);
            if(show){
                
                for(auto r : objects){
                    cv::rectangle(image, r, Scalar(0,0,255),2);
                }
                cv::imshow("src", image);
                cv::waitKey(0);
            }
        }
    }

    void ConvertICDAR2013::read_test_data(int show){
        for(auto d : data_->test){
            cv::Mat image = cv::imread(d.second.image_name);
            vector<Rect> objects;
            xml_->read_voc_format(d.second.annotation_name,objects);
            if(show){
                for(auto r : objects){
                    cv::rectangle(image, r, Scalar(0,0,255),2);
                }
                cv::imshow("src", image);
                cv::waitKey(0);
            }
        }
    }


    void ConvertICDAR2013::read_train(){
        ifstream fi(this->data_->path+"/"+"Splits/train.txt", ios::in);
        std::string name;
        while(fi >> name){
            DataPair data_pair;
            data_pair.image_name = this->data_->path+"/"+"PNGImages/"+name+".jpg";
            data_pair.annotation_name = this->data_->path+"/"+"Annotations/"+name+".xml";
            data_->train.insert(std::make_pair(name, data_pair));
        }
        fi.close();
    }

    void ConvertICDAR2013::read_test(){
        ifstream fi(this->data_->path+"/"+"Splits/test.txt", ios::in);
        std::string name;
        while(fi >> name){
            DataPair data_pair;
            data_pair.image_name = this->data_->path+"/"+"PNGImages/"+name+".jpg";
            data_pair.annotation_name = this->data_->path+"/"+"Annotations/"+name+".xml";
            data_->test.insert(std::make_pair(name, data_pair));
        }
        fi.close();
    }


}