#include <memory>
#include "random-forest/tree.hpp"
#include "random-forest/data.hpp"

using namespace kurff;

int main(){
    int number_feat_stage = 100;
    int max_depth = 10;
    std::shared_ptr<Tree> tree(new Tree(number_feat_stage,max_depth));
    std::shared_ptr<Data> data(new Data());
    string path = "/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/data/characters/";
    int number = 2510;
    data->load_image_data(path, number);
    
    tree->train(data.get());
    


    return 0;
}