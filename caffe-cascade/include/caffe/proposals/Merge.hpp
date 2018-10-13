#ifndef __MERGE_HPP__
#define __MERGE_HPP__
#include "proposals/Proposal.hpp"
#include "proposals/CannyProposal.hpp"
#include "proposals/MSERProposal.hpp"
#include "utils/utils.hpp"
#include <vector>
using namespace std;
// merge proposal using simple overlapping theshold
namespace kurff{
    class Merge{
        public:
            Merge(){
                LOG(INFO)<<"create Merge method";

            }
            ~Merge(){}
            
            void merge(const vector<Box>& proposal1, const vector<Box>& proposal2, 
            vector<Box>& proposal){
                proposal.clear();
                vector<int> index;
                overlap(proposal1, proposal2, 0.7, index);
                int cnt = -1;
                for(auto p : proposal1){
                    Box box;
                    ++ cnt;
                    if( index[cnt] != -1){
                        merge_box(p, proposal2[index[cnt]], box);
                    }
                    proposal.push_back(box);
                }
            }

            void simple_merge(const vector<Box>& proposal1, const vector<Box>& proposal2, 
            vector<Box>& proposal){
                vector<Box> proposal_temp;
                proposal_temp.clear();

                if(proposal1.size() ==0 && proposal2.size() !=0){
                    proposal_temp.insert(proposal_temp.end(), proposal2.begin(), proposal2.end());
                    proposal.clear();
                    proposal.insert(proposal.end(), proposal_temp.begin(), proposal_temp.end());

                    return;
                }
                else if(proposal2.size() ==0 && proposal1.size()!=0){
                    proposal_temp.insert(proposal_temp.end(), proposal1.begin(), proposal1.end());
                    proposal.clear();
                    proposal.insert(proposal.end(), proposal_temp.begin(), proposal_temp.end());
                    return;
                }else if(proposal2.size() ==0 && proposal1.size() == 0 ){
                    proposal.clear();
                    return;
                }else{
                    //LOG(INFO)<<"merge";
                    vector<int> index;
                    overlap(proposal1, proposal2, 0.3, index);
                    for(int i = 0; i < proposal1.size(); ++ i){
                        if(index[i] == -1){
                            proposal_temp.push_back(proposal1[i]);
                        }else{
                            proposal_temp.push_back(proposal1[i]);
                            proposal_temp.push_back(proposal2[index[i]]);
                        }
                    }

                }
                proposal.clear();
                proposal.insert(proposal.end(), proposal_temp.begin(), proposal_temp.end());

            }
        protected:


    };


}

#endif