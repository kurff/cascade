#include <memory>
//#include "random-forest/tree.hpp"
#include "random-forest/data.hpp"
#include "random-forest/hierarchical.hpp"

using namespace kurff;

int main(){
    int number_feat_stage = 100;
    int max_depth = 20;
    //std::shared_ptr<Tree> tree(new Tree(number_feat_stage,max_depth));
    std::shared_ptr<Hierarchical> hier(new Hierarchical(2));
    std::shared_ptr<Data> data(new Data());
    string path = "/media/kurff/d45400e1-76eb-453c-a31e-9ae30fafb7fd/data/characters/";
    int number = 105;
    data->load_image_data(path, number);
    
    hier->clustering(data.get());
    hier->save_graphvis("z.png");
    //tree->save_graphvis("x");


    return 0;
}