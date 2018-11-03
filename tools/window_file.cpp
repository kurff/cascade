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



DEFINE_string(root_path, "", "root path of voc file");
DEFINE_string(file_name,"","file name of voc format");
DEFINE_string(file_out,"","file out");

int main(int argc, char* argv[]){
    gflags::SetUsageMessage("convert window file from voc format to txt file\n"
    "useage:\n"
    "window_file --root_path=./ --file_name=trainval.txt");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(argc !=3){
        gflags::ShowUsageWithFlagsRestrict(argv[0],"tool/window_file");
        return 1;
    }

    string name;
    ifstream fi(FLAGS_file_name, std::ios::in);
    ifstream fo(FLAGS_file_out, std::ios::out);
    vector<Rect> rects;
    std::shared_ptr<XML<Rect> > xml(new XML<Rect>());
    while(fi >> name){
        xml->read_voc_format(root_path+"/"+name+".xml", rects);
        
        for(auto r : rects){
            
        }
        

    }

    fi.close();
    fo.close();




}