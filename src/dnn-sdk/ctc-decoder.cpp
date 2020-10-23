
#include <numeric>
#include <set>
#include <map>
#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include "ctc-decoder.h"
using namespace std;
namespace DNN {
    /*
     * My implementation of Prefix Search Decoding *
     * Reference from page 65 [Algorithm 7.1] at
     *      https://www.cs.toronto.edu/~graves/phd.pdf
     */
    bool PrefixSearchDecoding(const std::vector<std::vector<float>>& probs, std::vector<int>* best_path, float* final_prob){
        if(probs.empty()) {
            cerr << __FILE__ << ":" << __LINE__ << ": " << "warning: input probability matrix is empty ";
            return false;
        }
        size_t alphbet_size = probs[0].size() - 1; // index 0 for blank
        size_t T = probs.size();
        for(int i = 0; i < probs.size(); ++i){
            if(probs[i].size() != alphbet_size+1) {
                cerr << __FILE__ << ":" << __LINE__ << ": " << "warning: input probability is not a matrix";
                return false;
            }
            if(float(std::accumulate(probs[i].begin(), probs[i].end(), 0.0)) != float(1)) {
                cerr << __FILE__ << ":" << __LINE__ << ": " << "warning: at time " << i+1 << ", the sumation of probability is not 1";
                return false;
            }
        }
        map<pair<vector<int>, int>, float> p_b;  // for blank
        map<pair<vector<int>, int>, float> p_nb; // for non-blank
        float last = 1.0;
        for(int t = 1; t <= T; t++){
            p_nb[make_pair(vector<int>(), t)] = 0;
            p_b[make_pair(vector<int>(), t)] = last * probs[t-1][0];
            last = p_b[make_pair(vector<int>(), t)];
        }
        map<vector<int>, float> prob;
        prob.emplace(vector<int>(), p_b[make_pair(vector<int>(), T)]);
        
        map<vector<int>, float> prob_ext;
        prob_ext.emplace(vector<int>(), 1 - p_b[make_pair(vector<int>(), T)]);
        vector<int> l_star;
        vector<int> p_star;
        set<vector<int>> P = {{}};
        while(prob_ext[p_star] > prob[l_star]){
            float probRemaining = prob_ext[p_star];
            for(int k = 1; k <= alphbet_size; ++k){
                vector<int> p = p_star;
                p.push_back(k);
                if(p_star.empty())
                    p_nb[make_pair(p, 1)] = probs[0][k];
                else p_nb[make_pair(p, 1)] = 0;
                p_b[make_pair(p, 1)] = 0;
                float prefixProb = p_nb[make_pair(p, 1)];
                for(int t = 2; t <= T; ++t){
                    float newLabelProb = p_b[make_pair(p_star, t-1)];
                    if(!p_star.empty() && p_star.back() == k)
                        newLabelProb += 0;
                    else newLabelProb += p_nb[make_pair(p_star, t-1)];
                    p_nb[make_pair(p, t)] = probs[t-1][k] * (newLabelProb + p_nb[make_pair(p, t-1)]);
                    p_b[make_pair(p, t)] = probs[t-1][0] * (p_b[make_pair(p, t-1)] + p_nb[make_pair(p, t-1)]);
                    prefixProb += probs[t-1][k] * newLabelProb;
                }
                prob[p] = p_b[make_pair(p, T)] + p_nb[make_pair(p, T)];
                prob_ext[p] = prefixProb - prob[p];
                probRemaining -= prob_ext[p];
                if(prob[p] > prob[l_star]) {
                    l_star = p;
                }
                if(prob_ext[p] > prob[l_star]){
                    P.insert(p);
                }
                if(probRemaining <= prob[l_star]) {
                    break;
                }
            }
            P.erase(p_star);
            if(P.empty()) break;
            float max_pro = std::numeric_limits<float>::min();
            for(auto iter = P.begin(); iter != P.end(); ++iter){
                if(prob[*iter] > max_pro) {
                    max_pro = prob[*iter];
                    p_star = *iter;
                }
            }
        }
        if(final_prob){
            *final_prob = prob[l_star];
        }
        if(best_path){
            *best_path = l_star;
        }
        return true;
    }
    /*
     * My implementation of Beam Search Decoding *
     * Reference from Algorithm 1 CTC Beam Search at
     *      http://proceedings.mlr.press/v32/graves14.pdf
     */
    bool BeamSearchDecoding(const std::vector<std::vector<float>>& probs, std::vector<int>* best_path, float* final_prob, int beam){
        if(probs.empty()) {
            cerr << __FILE__ << ":" << __LINE__ << ": " << "warning: input probability matrix is empty ";
            return false;
        }
        size_t alphbet_size = probs[0].size() - 1; // index 0 for blank
        size_t T = probs.size();
        for(int i = 0; i < probs.size(); ++i){
            if(probs[i].size() != alphbet_size+1) {
                cerr << __FILE__ << ":" << __LINE__ << ": " << "warning: input probability is not a matrix";
                return false;
            }
            if(float(std::accumulate(probs[i].begin(), probs[i].end(), 0.0)) != float(1)) {
                cerr << __FILE__ << ":" << __LINE__ << ": " << "warning: at time " << i+1 << ", the sumation of probability is not 1";
                return false;
            }
        }
        set<vector<int>> B = { {} };
        map<pair<vector<int>, int>, float> p_b;  // for blank
        map<pair<vector<int>, int>, float> p_nb; // for non-blank
        float last = 1.0;
        p_nb[make_pair(vector<int>(), 0)] = 0;
        p_b[make_pair(vector<int>(), 0)] = last;
        for(int t = 1; t <= T; t++){
            p_nb[make_pair(vector<int>(), t)] = 0;
            p_b[make_pair(vector<int>(), t)] = last * probs[t-1][0];
            last = p_b[make_pair(vector<int>(), t)];
        }
        for(int t = 1; t <= T; ++t){
            set<vector<int>> B_hat;
            vector<pair<float,vector<int>>> q;
            for(auto iter = B.begin(); iter != B.end(); iter++){
                float p = p_b[make_pair(*iter, t-1)] + p_nb[make_pair(*iter, t-1)];
                q.push_back(make_pair(p, *iter));
            }
            sort(q.begin(), q.end(), [](const pair<float,vector<int>> &a, const pair<float,vector<int>> &b) {
                return (a.first > b.first);
            });
            for(int k = 0; k < ((q.size() < beam)? q.size() : beam); ++k){
                B_hat.insert(q[k].second);
            }
            
            B.clear();
            for(auto iter = B_hat.begin(); iter != B_hat.end(); iter++){
                if(!iter->empty()){
                    p_nb[make_pair(*iter, t)] = p_nb[make_pair(*iter, t-1)] * probs[t-1][iter->back()];
                    auto y_hat = *iter;
                    y_hat.pop_back();
                    if(B_hat.find(y_hat) != B_hat.end()){
                        if(!y_hat.empty() && y_hat.back() == iter->back()){
                            p_nb[make_pair(*iter, t)] += probs[t-1][iter->back()] * p_b[make_pair(y_hat, t-1)] ;
                        }else{
                            p_nb[make_pair(*iter, t)] += probs[t-1][iter->back()] * (p_b[make_pair(y_hat, t-1)] + p_nb[make_pair(y_hat, t-1)]);
                        }
                    }
                }
                p_b[make_pair(*iter, t)] = probs[t-1][0] * (p_nb[make_pair(*iter, t - 1)] + p_b[make_pair(*iter, t - 1)]);
                B.insert(*iter);
                for(int k = 1; k <= alphbet_size; ++k){
                    auto y_plus_k = *iter;
                    y_plus_k.push_back(k);
                    p_b[make_pair(y_plus_k, t)] = 0;
                    if(!iter->empty() && iter->back() == k){
                        p_nb[make_pair(y_plus_k, t)] = probs[t-1][k] * p_b[make_pair(*iter, t-1)];
                    }else{
                        p_nb[make_pair(y_plus_k, t)] = probs[t-1][k] * (p_b[make_pair(*iter, t-1)] + p_nb[make_pair(*iter, t-1)]);
                    }
                    B.insert(y_plus_k);
                }
            }
        }
        float max_pro = std::numeric_limits<float>::min();
        for(auto iter = B.begin(); iter != B.end(); iter++){
            float p = p_b[make_pair(*iter, T)] + p_nb[make_pair(*iter, T)];
            if(p > max_pro){
                max_pro = p;
                if(final_prob){
                    *final_prob = p;
                }
                if(best_path){
                    *best_path = *iter;
                }
            }
        }
        return true;
    }
}// end of namespace DNN
