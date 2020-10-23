

#include "rnn.h"
namespace DNN{
    RNN::RNN(int32_t i_dim, int32_t cell, ACTIVATION_TYPE a_t, bool r_seq, size_t offset_input): Layer(i_dim, cell, LayerType::RNN), ac_type(a_t), return_sequence(r_seq), offset_input_(offset_input){
        initial.Resize(1, OutputDim());

        activation = CreateActivation(a_t, cell);
    }
    bool RNN::GetReturnSeq() const{
        return return_sequence;
    }
    void RNN::SetReturnSeq(bool ret){
        return_sequence = ret;
    }
    void RNN::WriteData(std::ostream &os, bool binary) const{
        
        WriteToken(os, binary, "<Offset>");
        if (!binary) os << "\n";
        WriteBasicType(os, binary, offset_input_);
        
        WriteToken(os, binary, "<ReturnSeq>");
        if (!binary) os << "\n";
        WriteBasicType(os, binary, return_sequence);
        
        WriteToken(os, binary, "<W>");
        if (!binary) os << "\n";
        W.Write(os, binary);
        
        WriteToken(os, binary, "<U>");
        if (!binary) os << "\n";
        U.Write(os, binary);
        
        WriteToken(os, binary, "<B1>");
        if (!binary) os << "\n";
        B1.Write(os, binary);
        
        WriteToken(os, binary, "<B2>");
        if (!binary) os << "\n";
        B2.Write(os, binary);
        
        WriteToken(os, binary, "<Activation>");
        if(activation){
            activation->WriteData(os, binary);
        }else{
            WriteToken(os, binary, "NONE");
        }
        if (!binary) os << "\n";
        
    }
        
    void RNN::ReadData(std::istream &is, bool binary){
        
        ExpectToken(is, binary, "<Offset>");
        ReadBasicType(is, binary, &offset_input_);
        
        ExpectToken(is, binary, "<ReturnSeq>");
        ReadBasicType(is, binary, &return_sequence);
        // forget gate
        ExpectToken(is, binary, "<W>");
        W.Read(is, binary);
        ExpectToken(is, binary, "<U>");
        U.Read(is, binary);
        ExpectToken(is, binary, "<B1>");
        B1.Read(is, binary);

        ExpectToken(is, binary, "<B2>");
        B2.Read(is, binary);

        ExpectToken(is, binary, "<Activation>");
        std::string token;
        ReadToken(is, binary, &token);
        SetActivation(MarkerToACtype(token));
    }
    void RNN::SetCoef(const Matrix<float>& w, const Matrix<float>& u, const Vector<float>& b1, const Vector<float>& b2){
        W = w;
        U = u;
        B1 = b1;
        B2 = b2;

    }
    RNN::~RNN(){
        if(activation){
            delete activation;
            activation = nullptr;
        }
    }
    void RNN::Activate(MatrixBase<float>* in, MatrixBase<float> *out, std::vector<int>* target_frames) const{
        if(activation){
            activation->PropagateFnc(in, out, target_frames);
        }else {
            out->Swap(in);
            LOG_WARN("Note: No activation function for " << this->GetIndex() << "th layer, if you set the layer without the activation function, it may go wrong except for the last layer!");
        }
    }
    void RNN::SetActivation(ACTIVATION_TYPE ac){
        ac_type = ac;
        activation = CreateActivation(ac, OutputDim());
    }
    void RNN::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                            std::vector<int>* target_frames) const{
        LOG_ASSERT(in->NumCols() == _input_dim);
        LOG_ASSERT(out->NumCols() == _output_dim);
        auto time_steps = in->NumRows();
        if(return_sequence) {
            LOG_ASSERT(out->NumRows() == time_steps - offset_input_);
        }else {
            LOG_ASSERT(out->NumRows() == 1);
        }
        Matrix<float> temp_out1;
        temp_out1.Resize(1, _output_dim);
        Matrix<float> temp_out2;
        temp_out2.Resize(1, _output_dim);
        for(int t = 0; t < time_steps - offset_input_; ++t){
            // input at time t
            SubMatrix<float> input_t(*in, t, 1, 0, _input_dim);
            
            int row_offset_last = (!return_sequence) ? 0 : (t == 0 ? 0 : t-1);
            int row_offset = (!return_sequence) ? 0 : t;
            
            SubMatrix<float> output_t_last(*out, row_offset_last, 1, 0, _output_dim);
            SubMatrix<float> output_t(*out, row_offset, 1, 0, _output_dim);
            
            AddMatMat(temp_out1, input_t, W);
            AddBias(temp_out1, B1);
            
            if(t == 0){
                AddMatMat(temp_out2, initial, U);
            }else{
                AddMatMat(temp_out2, output_t_last, U);
            }
            AddBias(temp_out2, B2);
            
            AddMat(temp_out1, temp_out2);
            Activate(&temp_out1, &output_t);
        }
        
    }
    
}
