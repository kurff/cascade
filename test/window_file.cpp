#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "tools/xml.hpp"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
using namespace kurff;



DEFINE_string(root_path, "/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/data/ICDAR2013_VOC/", "root path of voc file");
DEFINE_string(file_name,"/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/data/ICDAR2013_VOC/Splits/train.txt","file name of voc format");
DEFINE_string(file_out,"/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/data/ICDAR2013_VOC/window_icdar2013.txt","file out");
DEFINE_string(file_size,"/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/data/ICDAR2013_VOC/icdar2013_size.txt","file out");


int main(int argc, char* argv[]){
    gflags::SetUsageMessage("convert window file from voc format to txt file\n"
    "useage:\n"
    "window_file --root_path=./ --file_name=trainval.txt");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // if(argc !=3){
    //     gflags::ShowUsageWithFlagsRestrict(argv[0],"tool/window_file");
    //     return 1;
    // }

    string name;
    ifstream fi(FLAGS_file_name, std::ios::in);
    ofstream fo(FLAGS_file_out, std::ios::out);
    ofstream f(FLAGS_file_size, std::ios::out);
    vector<Rect> rects;
    std::shared_ptr<XML<Rect> > xml(new XML<Rect>());
    int cnt = -1;
    while(fi >> name){
        LOG(INFO)<<"name: "<< name;
        xml->read_voc_format(FLAGS_root_path+"/Annotations/"+name+".xml", rects);
        ++ cnt;
        fo << "# "<<cnt<<std::endl;
        string image_name ="/PNGImages/"+name+".jpg";
        fo << image_name << std::endl;
        cv::Mat image = cv::imread(FLAGS_root_path+"/"+image_name);
        f << image_name<<" "<< image.rows <<" "<< image.cols << std::endl;
        fo << image.channels() << std::endl;
        fo << image.rows << std::endl;
        fo << image.cols << std::endl;
        fo << rects.size() << std::endl;
        for(auto r : rects){
            fo <<"1 0 0 "<< r.x <<" "<<r.y <<" "<<r.x + r.width<<" "<<r.y + r.height<<std::endl;
        }
        fo <<"0"<<std::endl;
    }


    f.close();
    fi.close();
    fo.close();
}