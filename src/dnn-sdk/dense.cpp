


#include "dense.h"
#include <fstream>
namespace DNN{

    Dense::Dense(ACTIVATION_TYPE a_t) : Layer(LayerType::DENSE), ac_type(a_t){
        activation = CreateActivation(a_t);
    }
    Dense::Dense(int32_t i_dim, int32_t o_dim, ACTIVATION_TYPE a_t) : Layer(i_dim, o_dim, LayerType::DENSE), ac_type(a_t){
        weights.Resize(o_dim, i_dim);
        bias.Resize(o_dim);
        activation = CreateActivation(a_t, o_dim);
    }
    Dense::Dense(const Matrix<float>& W, const Vector<float>& b, ACTIVATION_TYPE a_t): Layer(W.NumCols(), W.NumRows(), LayerType::DENSE), weights(W), bias(b), ac_type(a_t){
        activation = CreateActivation(a_t, bias.Dim());

    }
    Dense::~Dense(){
        if(activation){
            delete activation;
            activation = nullptr;
        }
    }
    void Dense::WriteData(std::ostream &os, bool binary) const{
        
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
    void Dense::ReadData(std::istream &is, bool binary){
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
    void Dense::Activate(MatrixBase<float>* in, MatrixBase<float> *out,
                         std::vector<int>* target_frames) const {
        if(activation){
            activation->PropagateFnc(in, out, target_frames);
        }else {
            out->Swap(in);
        }
    }
    void Dense::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                             std::vector<int>* target_frames) const {
        AddMatMat(*out, *in, weights, target_frames);
        AddBias(*out, bias, target_frames);
        Activate(out, out, target_frames);
    }
    void Dense::SetActivation(ACTIVATION_TYPE ac){
        ac_type = ac;
        activation = CreateActivation(ac, OutputDim());
    }

}// end of namespace DNN

