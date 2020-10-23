

#include <stdio.h>
#include "tdnn.h"

namespace DNN{

    Tdnn::Tdnn(int32_t i_dim, int32_t o_dim,
         ACTIVATION_TYPE a_t,
         bool norm_) : Layer(i_dim, o_dim, LayerType::TDNN),
        ac_type(a_t), norm(norm_){
        activation = CreateActivation(a_t, o_dim);
    }
    Tdnn::Tdnn(int32_t i_dim, int32_t o_dim,
               const std::vector<int32_t>& input_context,
               ACTIVATION_TYPE a_t,
               bool norm_) : Layer(i_dim, o_dim, input_context, LayerType::TDNN), ac_type(a_t), norm(norm_){
        activation = CreateActivation(a_t, o_dim);
    }
    Tdnn::Tdnn(const Matrix<float>& weights_,
               const Vector<float>& bias_,
               const std::vector<int32_t>& input_context,
               ACTIVATION_TYPE a_t,
               bool norm_): Layer(weights_.NumCols(), weights_.NumRows(), input_context, LayerType::TDNN),  weights(weights_), bias(bias_), ac_type(a_t), norm(norm_){
        LOG_ASSERT(weights_.NumRows() == bias_.Dim());
        activation = CreateActivation(a_t, weights_.NumRows());

    }
    Tdnn::~Tdnn(){
        if(activation){
            delete activation;
            activation = nullptr;
        }
    }
    void Tdnn::WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<ReNorm>");
        if (!binary) os << "\n";
        WriteBasicType(os, binary, norm);
        
        WriteToken(os, binary, "<Context>");
        if (!binary) os << "\n";
        Vector<int32_t> context(Context());
        context.Write(os, binary);
        
        WriteToken(os, binary, "<Weights>");
        if (!binary) os << "\n";
        weights.Write(os, binary);
        
        WriteToken(os, binary, "<Bias>");
        if (!binary) os << "\n";
        bias.Write(os, binary);
        
        WriteToken(os, binary, "<Activation>");
        if(activation){
            activation->WriteData(os, binary);
        }else{
            WriteToken(os, binary, "NONE");
        }
        if (!binary) os << "\n";
    }
    void Tdnn::ReadData(std::istream &is, bool binary){
        ExpectToken(is, binary, "<ReNorm>");
        ReadBasicType(is, binary, &norm);
        
        ExpectToken(is, binary, "<Context>");
        Vector<int32_t> context;
        context.Read(is, binary);
        std::vector<int32_t> cont(context.Dim());
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMemcpy(cont.data(), context.Data(), context.Dim()*sizeof(int32_t), cudaMemcpyDeviceToHost));
#else
        for(int i = 0; i < context.Dim(); ++i){
            cont[i] = context(i);
        }
#endif
        SetContext(cont);
        
        ExpectToken(is, binary, "<Weights>");
        weights.Read(is, binary);
        
        ExpectToken(is, binary, "<Bias>");
        bias.Read(is, binary);

        ExpectToken(is, binary, "<Activation>");
        std::string token;
        ReadToken(is, binary, &token);
        SetActivation(MarkerToACtype(token));
    }
    void Tdnn::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                            std::vector<int>* target_frames) const{
        {
            AddMatMat(*out, *in, weights, input_context, target_frames);
        }
        {
            AddBias(*out, bias, target_frames);
        }
        {
            Activate(out, out, target_frames);
        }
        if(norm){
            Normalization(*out, *out, target_frames);
        }
    }
    void Tdnn::Activate(MatrixBase<float>* in, MatrixBase<float> *out, std::vector<int>* target_frames) const{
        if(activation){
            activation->PropagateFnc(in, out, target_frames);
        }else {
            out->Swap(in);
            LOG_WARN("Note: No activation function for " << this->GetIndex() << "th layer, if you set the layer without the activation function, it may go wrong except for the last layer!");
        }
    }
    void Tdnn::SetActivation(ACTIVATION_TYPE ac){
        ac_type = ac;
        activation = CreateActivation(ac, OutputDim());
    }

} // namespace DNN
