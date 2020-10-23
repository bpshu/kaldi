

#include "bi-rnn.h"
namespace DNN{
    BRNN::BRNN(int32_t i_dim, int32_t cell, ACTIVATION_TYPE a_t, size_t offset_input): Layer(i_dim, cell * 2, LayerType::BRNN), offset_input_(offset_input) {
        activation = CreateActivation(a_t, cell * 2);
        initial.Resize(1, cell);

    }

    void BRNN::WriteData(std::ostream &os, bool binary) const{
        
        WriteToken(os, binary, "<Offset>");
        if (!binary) os << "\n";
        WriteBasicType(os, binary, offset_input_);
        
        WriteToken(os, binary, "<W_fwd>");
        if (!binary) os << "\n";
        W_fwd.Write(os, binary);
        
        WriteToken(os, binary, "<U_fwd>");
        if (!binary) os << "\n";
        U_fwd.Write(os, binary);
        
        WriteToken(os, binary, "<B1_fwd>");
        if (!binary) os << "\n";
        B1_fwd.Write(os, binary);
    
        WriteToken(os, binary, "<B2_fwd>");
        if (!binary) os << "\n";
        B2_fwd.Write(os, binary);
        
        // backword
        WriteToken(os, binary, "<W_back>");
        if (!binary) os << "\n";
        W_back.Write(os, binary);
        
        WriteToken(os, binary, "<U_back>");
        if (!binary) os << "\n";
        U_back.Write(os, binary);
        
        WriteToken(os, binary, "<B1_back>");
        if (!binary) os << "\n";
        B1_back.Write(os, binary);
        
        WriteToken(os, binary, "<B2_back>");
        if (!binary) os << "\n";
        B2_back.Write(os, binary);
        
        WriteToken(os, binary, "<Activation>");
        if(activation){
            activation->WriteData(os, binary);
        }else{
            WriteToken(os, binary, "NONE");
        }
        if (!binary) os << "\n";
        
    }
    
    void BRNN::ReadData(std::istream &is, bool binary){
        ExpectToken(is, binary, "<Offset>");
        ReadBasicType(is, binary, &offset_input_);
        
        ExpectToken(is, binary, "<W_fwd>");
        W_fwd.Read(is, binary);
        ExpectToken(is, binary, "<U_fwd>");
        U_fwd.Read(is, binary);
        ExpectToken(is, binary, "<B1_fwd>");
        B1_fwd.Read(is, binary);
        ExpectToken(is, binary, "<B2_fwd>");
        B2_fwd.Read(is, binary);
        
        ExpectToken(is, binary, "<W_back>");
        W_back.Read(is, binary);
        ExpectToken(is, binary, "<U_back>");
        U_back.Read(is, binary);
        ExpectToken(is, binary, "<B1_back>");
        B1_back.Read(is, binary);
        ExpectToken(is, binary, "<B2_back>");
        B2_back.Read(is, binary);

        ExpectToken(is, binary, "<Activation>");
        std::string token;
        ReadToken(is, binary, &token);
        SetActivation(MarkerToACtype(token));
    }
    void BRNN::SetCoef(const Matrix<float>& w_fwd, const Matrix<float>& u_fwd, const Vector<float>& b1_fwd, const Vector<float>& b2_fwd, const Matrix<float>& w_back, const Matrix<float>& u_back, const Vector<float>& b1_back, const Vector<float>& b2_back){
        W_fwd = w_fwd;
        U_fwd = u_fwd;
        B1_fwd = b1_fwd;
        B2_fwd = b2_fwd;
        
        W_back = w_back;
        U_back = u_back;
        B1_back = b1_back;
        B2_back = b2_back;

    }
    BRNN::~BRNN(){
        if(activation){
            delete activation;
            activation = nullptr;
        }
    }
    void BRNN::Activate(MatrixBase<float>* in, MatrixBase<float> *out, std::vector<int>* target_frames) const{
        if(activation){
            activation->PropagateFnc(in, out, target_frames);
        }else {
            out->Swap(in);
            LOG_WARN("Note: No activation function for " << this->GetIndex() << "th layer, if you set the layer without the activation function, it may go wrong except for the last layer!");
        }
    }
    void BRNN::SetActivation(ACTIVATION_TYPE ac){
        ac_type = ac;
        activation = CreateActivation(ac, OutputDim());
    }
    void BRNN::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                           std::vector<int>* target_frames) const{
        LOG_ASSERT(in->NumCols() == _input_dim);
        LOG_ASSERT(out->NumCols() == _output_dim);
        auto time_steps = in->NumRows();
        LOG_ASSERT(out->NumRows() == time_steps - offset_input_);
        
        Matrix<float> temp_out1;
        temp_out1.Resize(1, _output_dim/2);
        Matrix<float> temp_out2;
        temp_out2.Resize(1, _output_dim/2);
        
        //forward pass hidden layer
        for(int t = 0; t < time_steps - offset_input_; ++t){
            // input at time t
            SubMatrix<float> input_t(*in, t, 1, 0, _input_dim);
            // output at time t
            int row_offset = (t == 0 ? 0 : t - 1);
            SubMatrix<float> fwd_out_t_last(*out, row_offset, 1, 0, _output_dim/2);
            SubMatrix<float> fwd_out_t(*out, t, 1, 0, _output_dim/2);
            
            AddMatMat(temp_out1, input_t, W_fwd);
            AddBias(temp_out1, B1_fwd);
            
            if(t == 0){
                AddMatMat(temp_out2, initial, U_fwd);
            }else{
                AddMatMat(temp_out2, fwd_out_t_last, U_fwd);
            }
            AddBias(temp_out2, B2_fwd);
            
            AddMat(temp_out1, temp_out2);
            Activate(&temp_out1, &fwd_out_t);
        }
        
        //backward pass hidden layer
        for(int t = time_steps - 1; t >= static_cast<int>(offset_input_); --t){
            // input at time t
            SubMatrix<float> input_t(*in, t, 1, 0, _input_dim);
            
            // output at time t
            int row_offset = (t == (time_steps - 1) ? (time_steps - 1) : t + 1);
            SubMatrix<float> back_out_t_last(*out, row_offset, 1, _output_dim/2, _output_dim/2);
            SubMatrix<float> back_out_t(*out, t, 1, _output_dim/2, _output_dim/2);
            AddMatMat(temp_out1, input_t, W_back);
            AddBias(temp_out1, B1_back);
            if(t == time_steps - 1){
                AddMatMat(temp_out2, initial, U_back);
            }else{
                AddMatMat(temp_out2, back_out_t_last, U_back);
            }
            AddBias(temp_out2, B2_back);
            
            AddMat(temp_out1, temp_out2);
            Activate(&temp_out1, &back_out_t);
        }
    }

}
