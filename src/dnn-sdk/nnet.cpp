

#include "nnet.h"
#include <set>
#include <sstream>
using namespace std;
namespace DNN{
    std::atomic<int> Nnet::subsamp(1);
    int Nnet::Subsample(){
        return subsamp.load();
    }
    void Nnet::SetSubsample(int i){
        subsamp.store(i);
    }
    std::string Nnet::NnetInfo() const{
        ostringstream ss;
        ss << "Number of Layers: " << NumComponents() << endl;
        ss << "Subsampling factor: " << Nnet::Subsample() << endl;
        if(NumComponents() > 0){
            ss << "Net input dim is " << GetLayer(0)->InputDim() << " , output dim is " <<  GetLayer(NumComponents() - 1)->OutputDim() << endl;
            ss << "Layer infomation is below: " << endl;
            for(int i = 0; i < NumComponents(); ++i){
                auto context = GetLayer(i)->Context();
                string con = "[ ";
                for(auto x : context) con += to_string(x) + " ";
                con += "] ";
                ss << i+1 << ". " << GetLayer(i)->TypeToMarker(GetLayer(i)->GetLayerType()) << ": input dim is "<< GetLayer(i)->InputDim() << ", outputdim is " << GetLayer(i)->OutputDim() << ", Context is " << con << endl;
            }
        }
        return ss.str();
    }
    Nnet::Nnet(): left_context(0), right_context(0){}
    Nnet::~Nnet(){
        Release();
    }
    void Nnet::Release(){
        for(auto p : _layers){
            if(p){
                delete p;
                p = NULL;
            }
        }
        _layers.clear();
        priors.clear();
        left_context = 0;
        right_context = 0;
    }
    void Nnet::GetFramesToCalculate(size_t frames,
                                    std::vector<std::vector<int>>& frames_to_cal) const{
        size_t nlayers = NumComponents();
        frames_to_cal.clear();
        frames_to_cal.resize(nlayers + 1);
        
        for(int layer = (int)nlayers; layer >= 0; --layer){
            if(layer == (int)nlayers){
                // the output frames
                auto actual_frames = (frames + Subsample() - 1) / Subsample();
                frames_to_cal[layer].reserve(actual_frames);
                for(int i = 0; i < actual_frames; ++i){
                    frames_to_cal[layer].emplace_back(i * Subsample());
                }
            }else{
                std::set<int> s; // sort and unique
                for(auto frame : frames_to_cal[layer+1]){
                    for(auto c : _layers[layer]->Context()){
                        s.emplace(frame + c);
                    }
                }
                frames_to_cal[layer] = std::vector<int>(s.begin(), s.end());
            }
        }
        size_t left, right;
        for(int i = 0; i < frames_to_cal.size() - 1; ++i){
            GetContextFromLayer(i, left, right);
            for(auto &tmp : frames_to_cal[i]){
                tmp += left;
            }
        }
    }

    void Nnet::Write(std::ostream &os, bool binary) const{
        WriteToken(os, binary, "<Nnet>");
        if (!binary) os << std::endl;
        for (int32_t i = 0; i < NumComponents(); i++) {
            _layers[i]->Write(os, binary);
        }
        WriteToken(os, binary, "</Nnet>");
        if (!binary) os << std::endl;
        WriteToken(os, binary, "<Prior>");
        if (!binary) os << std::endl;
        Vector<float> p(priors);
        p.Write(os, binary);
        WriteToken(os, binary, "</Prior>");
        if (!binary) os << std::endl;
    }
    void Nnet::Read(std::istream &is, bool binary){
        Release();
        ExpectToken(is, binary, "<Nnet>");
        Layer* layer = nullptr;
        while(layer = Layer::Read(is, binary), layer != nullptr){
            AddLayer(layer);
        }
        if(is){
            ExpectToken(is, binary, "<Prior>");
            Vector<float> p(priors);
            p.Read(is, binary);
            std::vector<float> pri(p.Dim());
#ifdef USE_CUDA
            CU_SAFE_CALL(cudaMemcpy(pri.data(), p.Data(), p.Dim()*sizeof(float), cudaMemcpyDeviceToHost));
#else
            for(int i = 0; i < pri.size(); ++i){
                pri[i] = p(i);
            }
#endif
            SetPrior(pri);
            ExpectToken(is, binary, "</Prior>");
        }
    }
    const std::vector<Layer*>& Nnet::GetLayers() const {
        return _layers;
        
    }
    std::vector<Layer*>& Nnet::GetLayers() {
        return _layers;
        
    }
    const int32_t Nnet::NumComponents() const {
        return static_cast<int32_t>(_layers.size());
    }
    void Nnet::AddLayer(Layer* layer){
        if(!_layers.empty()){
            auto times = layer->Context().empty() ? 1 : layer->Context().size();
            LOG_ASSERT(_layers.back()->OutputDim() * times == layer->InputDim());
        }
        _layers.push_back(layer);
        if(!layer->Context().empty()){
            left_context += abs(layer->Context().front());
            right_context += abs(layer->Context().back());
        }
        layer->SetIndex(NumComponents());
    }
    void Nnet::RemoveLastLayer(){
        if(_layers.empty()) return;
        Layer* layer = _layers.back();
        _layers.pop_back();
        if(layer){
            delete layer;
            layer = NULL;
        }
    }
    const std::vector<float>& Nnet::GetPrior() const{
        return priors;
    }
    void Nnet::SetPrior(const std::vector<float>& p){
        priors = p;
    }
    Layer* Nnet::GetLayer(int32_t index) const{
        LOG_ASSERT(index >= 0 && index < this->NumComponents());
        return _layers[index];
    }
    void Nnet::GetContextFromLayer(int layer, size_t &left, size_t& right) const{
        LOG_ASSERT(layer >= 0 && layer < NumComponents());
        left = 0;
        right = 0;
        for(int i = layer; i < NumComponents(); ++i){
            auto layer = _layers[i];
            if(!layer->Context().empty()){
                left += abs(layer->Context().front());
                right += abs(layer->Context().back());
            }
        }
    }
    void Nnet::GetContext(size_t &left, size_t& right) const {
        left = left_context;
        right = right_context;
    }
#ifdef QUANTIZATION
    void Nnet::FeedForward(Matrix<float> &in, Matrix<float> *out,
                           std::vector<std::vector<int>>* frames_to_cal) const{
        LOG_ASSERT(NULL != out);
        LOG_ASSERT(NumComponents() >= 2 && "No need to #define QUANTIZATION");
        Matrix<unsigned char> tmp_in; // output of the first layer
        if(frames_to_cal != nullptr){
            LOG_ASSERT(frames_to_cal->size() > 1);
            FeedForwardFirstLayer(in, &tmp_in, &(*frames_to_cal)[1]);
        }else{
            FeedForwardFirstLayer(in, &tmp_in);
        }
        Matrix<unsigned char> tmp_out = tmp_in;
        for(int32_t i = 1; i < NumComponents() - 1; ++i ){
            tmp_out.Swap(&tmp_in);
            if(frames_to_cal){
                LOG_ASSERT(frames_to_cal->size() > i+1);
                _layers[i]->Propagate(tmp_in, &tmp_out, &(*frames_to_cal)[i+1]);
            }else{
                _layers[i]->Propagate(tmp_in, &tmp_out);
            }
        }
        if(frames_to_cal){
            LOG_ASSERT(frames_to_cal->size() > NumComponents());
            _layers[NumComponents() - 1]->Propagate(tmp_out, out, &(*frames_to_cal)[NumComponents()]); // last
        }else{
            _layers[NumComponents() - 1]->Propagate(tmp_out, out); // last
        }
    }
#elif (! defined QUANTIZATION && ! defined LAZY)
    // only SSE_USE maybe?
    void Nnet::FeedForward(Matrix<float> &in, Matrix<float> *out,
                           std::vector<std::vector<int>>* frames_to_cal) const{
        LOG_ASSERT(nullptr != out);
        out->Resize(0, 0);
        (*out) = in;  // works even with 0 components
        Matrix<float> tmp_in;
        for (int32_t i = 0; i < NumComponents(); i++) {
            out->Swap(&tmp_in);
            if(frames_to_cal){
                LOG_ASSERT(frames_to_cal->size() > i+1);	
                _layers[i]->Propagate(tmp_in, out, &(*frames_to_cal)[i+1]);
            }else{
                _layers[i]->Propagate(tmp_in, out);
            }
        }
    }
#endif
    
#if(defined QUANTIZATION && defined LAZY)
    void Nnet::FeedForward(Matrix<float> &in, Matrix<unsigned char> *out,
                           std::vector<std::vector<int>>* frames_to_cal) const {
        LOG_ASSERT(nullptr != out);
        Matrix<unsigned char> tmp_in; // output of the first layer
        if(frames_to_cal != nullptr){
            LOG_ASSERT(frames_to_cal->size() > 1);
            FeedForwardFirstLayer(in, &tmp_in, &(*frames_to_cal)[1]);
        }else{
            FeedForwardFirstLayer(in, &tmp_in);
        }
        //(*out) = tmp_in;
        for(int32_t i = 1; i < NumComponents() - 1 /* here -1 is for lazy implementation */;
            ++i ){
            //out->Swap(&tmp_in);
            if(frames_to_cal){
                LOG_ASSERT(frames_to_cal->size() > i+1);
                _layers[i]->Propagate(tmp_in, out, &(*frames_to_cal)[i+1]);
            }else{
                _layers[i]->Propagate(tmp_in, out);
            }
	     if(i != NumComponents() - 2)
                out->Swap(&tmp_in);
        }
    }
    void Nnet::FeedForwardLastNoQuantize(Matrix<float> &in, Matrix<float> *out,
                           std::vector<std::vector<int>>* frames_to_cal) const {
        LOG_ASSERT(nullptr != out);
        LOG_ASSERT(NumComponents() > 1);
        if(NumComponents() == 2){
            if(frames_to_cal){
                LOG_ASSERT(frames_to_cal->size() > 2);
                _layers[0]->Propagate(in, out, &(*frames_to_cal)[1]);
            }else{
                _layers[0]->Propagate(in, out);
            }
        }else{ // NumComponents() > 2
            Matrix<unsigned char> tmp_in; // output of the first layer
            if(frames_to_cal != nullptr){
                LOG_ASSERT(frames_to_cal->size() > 1);
                FeedForwardFirstLayer(in, &tmp_in, &(*frames_to_cal)[1]);
            }else{
                FeedForwardFirstLayer(in, &tmp_in);
            }
            Matrix<unsigned char> tmp_out;
            for(int32_t i = 1; i < NumComponents() - 1 /* here -1 is for lazy implementation */;
                ++i ){
                if(i == NumComponents() - 2){
                    if(frames_to_cal){
                        LOG_ASSERT(frames_to_cal->size() > i+1);
                        _layers[i]->Propagate(tmp_in, out, &(*frames_to_cal)[i+1]);
                    }else{
                        _layers[i]->Propagate(tmp_in, out);
                    }
                }else{
                    if(frames_to_cal){
                        LOG_ASSERT(frames_to_cal->size() > i+1);
                        _layers[i]->Propagate(tmp_in, &tmp_out, &(*frames_to_cal)[i+1]);
                    }else{
                        _layers[i]->Propagate(tmp_in, &tmp_out);
                    }
                    tmp_out.Swap(&tmp_in);
                }
            }
        }
    }
#endif
#ifdef QUANTIZATION
    void Nnet::FeedForwardFirstLayer(Matrix<float> &in, Matrix<unsigned char> *out,
                                     std::vector<int>* target_frames) const{
        LOG_ASSERT(NULL != out);
        if(NumComponents() <= 0){
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << "Number of DNN layers is zero, can not feed forwarding."<< std::endl;
            exit(1);
        }
        _layers[0]->PropagateFirstOnly(in, out, target_frames); // only propagate the first layer
    }
#endif
    
}// end of namespace DNN
