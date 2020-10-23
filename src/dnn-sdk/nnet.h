

#ifndef nnet_h
#define nnet_h
#include <vector>
#include <atomic>
#include "layer.h"
namespace DNN{
class Nnet{
    public:
        Nnet();
        ~Nnet();
        std::string NnetInfo() const;
        void GetFramesToCalculate(size_t frames,
                                  std::vector<std::vector<int>>& frames_to_cal) const;
        void Write(std::ostream &os, bool binary) const;
        void Read(std::istream &is, bool binary);
        const std::vector<Layer*>& GetLayers() const;
        std::vector<Layer*>& GetLayers();
        const int32_t NumComponents() const;
        void AddLayer(Layer* layer);
        void RemoveLastLayer();
        const std::vector<float>& GetPrior() const;
        void SetPrior(const std::vector<float>& p);
        Layer* GetLayer(int32_t index) const;
        void GetContext(size_t &left, size_t& right) const;
        void GetContextFromLayer(int layer, size_t &left, size_t& right) const;
        virtual void FeedForward(Matrix<float> &in, Matrix<float> *out,
                                 std::vector<std::vector<int>>* frames_to_cal = nullptr ) const;
        void Release();
        static int Subsample();
        static void SetSubsample(int i);

    protected:
        static std::atomic<int> subsamp;
        std::vector<Layer*> _layers;
        size_t left_context;
        size_t right_context;
        std::vector<float> priors;

    };
}// end of namespace DNN

#endif /* nnet_hpp */
