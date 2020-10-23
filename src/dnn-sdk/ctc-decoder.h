

#ifndef ctc_decoder_h
#define ctc_decoder_h

#include <vector>
namespace DNN{
    bool PrefixSearchDecoding(const std::vector<std::vector<float>>& probs, std::vector<int>* best_path, float* final_prob);

    bool BeamSearchDecoding(const std::vector<std::vector<float>>& probs, std::vector<int>* best_path, float* final_prob, int beam);
} // end of namespace DNN
#endif /* ctc_decoder_h */
