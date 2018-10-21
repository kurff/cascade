#include "tools/convert.hpp"
#include "tools/convert_icdar2013.hpp"

using namespace kurff;
int main(int argc, char* argv[]){
    string path = "/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/data/ICDAR2013_VOC/";
    std::shared_ptr<Convert> convert(new ConvertICDAR2013(path, 300,300));
    convert->read_train();
    convert->write_train_data();


}