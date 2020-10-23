

#include "lstm.h"
namespace DNN{
    LSTM::LSTM(int32_t i_dim, int32_t cell, bool r_seq): Layer(i_dim, cell, LayerType::LSTM),  return_sequence(r_seq) {}
    bool LSTM::GetReturnSeq() const{
        return return_sequence;
    }
    void LSTM::SetReturnSeq(bool ret){
        return_sequence = ret;
    }
    void LSTM::WriteData(std::ostream &os, bool binary) const{
        WriteToken(os, binary, "<ReturnSeq>");
        if (!binary) os << "\n";
        WriteBasicType(os, binary, return_sequence);
        
        // forget gate
        WriteToken(os, binary, "<WF>");
        if (!binary) os << "\n";
        Wf.Write(os, binary);

        WriteToken(os, binary, "<UF>");
        if (!binary) os << "\n";
        Uf.Write(os, binary);
        
        WriteToken(os, binary, "<BF>");
        if (!binary) os << "\n";
        Bf.Write(os, binary);
        
        //input gate
        WriteToken(os, binary, "<WI>");
        if (!binary) os << "\n";
        Wi.Write(os, binary);
        
        WriteToken(os, binary, "<UI>");
        if (!binary) os << "\n";
        Ui.Write(os, binary);
        
        WriteToken(os, binary, "<BI>");
        if (!binary) os << "\n";
        Bi.Write(os, binary);
        
        //cell gate
        WriteToken(os, binary, "<WC>");
        if (!binary) os << "\n";
        Wc.Write(os, binary);
        
        WriteToken(os, binary, "<UC>");
        if (!binary) os << "\n";
        Uc.Write(os, binary);
        
        WriteToken(os, binary, "<BC>");
        if (!binary) os << "\n";
        Bc.Write(os, binary);
        
        //Output gate
        WriteToken(os, binary, "<WO>");
        if (!binary) os << "\n";
        Wo.Write(os, binary);
        
        WriteToken(os, binary, "<UO>");
        if (!binary) os << "\n";
        Uo.Write(os, binary);
        
        WriteToken(os, binary, "<BO>");
        if (!binary) os << "\n";
        Bo.Write(os, binary);
    }

    void LSTM::ReadData(std::istream &is, bool binary){
        ExpectToken(is, binary, "<ReturnSeq>");
        ReadBasicType(is, binary, &return_sequence);
        
        // forget gate
        ExpectToken(is, binary, "<WF>");
        Wf.Read(is, binary);
        ExpectToken(is, binary, "<UF>");
        Uf.Read(is, binary);
        ExpectToken(is, binary, "<BF>");
        Bf.Read(is, binary);
        
        // input gate
        ExpectToken(is, binary, "<WI>");
        Wi.Read(is, binary);
        ExpectToken(is, binary, "<UI>");
        Ui.Read(is, binary);
        ExpectToken(is, binary, "<BI>");
        Bi.Read(is, binary);
        
        // cell gate
        ExpectToken(is, binary, "<WC>");
        Wc.Read(is, binary);
        ExpectToken(is, binary, "<UC>");
        Uc.Read(is, binary);
        ExpectToken(is, binary, "<BC>");
        Bc.Read(is, binary);
        
        // output gate
        ExpectToken(is, binary, "<WO>");
        Wo.Read(is, binary);
        ExpectToken(is, binary, "<UO>");
        Uo.Read(is, binary);
        ExpectToken(is, binary, "<BO>");
        Bo.Read(is, binary);
        

        
    }
    void LSTM::SetCoef(const Matrix<float>& wf, const Matrix<float>& uf, const Vector<float>& bf,
                       const Matrix<float>& wi, const Matrix<float>& ui, const Vector<float>& bi,
                       const Matrix<float>& wc, const Matrix<float>& uc, const Vector<float>& bc,
                       const Matrix<float>& wo, const Matrix<float>& uo, const Vector<float>& bo){
        Wf = wf; Uf = uf; Bf = bf;
        Wi = wi; Ui = ui; Bi = bi;
        Wc = wc; Uc = uc; Bc = bc;
        Wo = wo; Uo = uo; Bo = bo;

    }
    LSTM::~LSTM(){}
    void LSTM::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                            std::vector<int>* target_frames) const{
        LOG_ASSERT(in->NumCols() == _input_dim);
        LOG_ASSERT(out->NumCols() == _output_dim);
        auto time_steps = in->NumRows();
        if(return_sequence) {
            LOG_ASSERT(out->NumRows() == time_steps);
        }else {
            LOG_ASSERT(out->NumRows() == 1);
        }
        Matrix<float> cell_state;   // cell state at time t
        cell_state.Resize(1, _output_dim); // initialize with all zero
        
//        Matrix<float> cell_output; // output at time t
//        cell_output.Resize(1, _output_dim); // initialize with all zero
        
        Matrix<float> temp_out1;
        temp_out1.Resize(1, _output_dim);
        Matrix<float> temp_out2;
        temp_out2.Resize(1, _output_dim);
        Matrix<float> it; // input
        it.Resize(1, _output_dim);
        Matrix<float> ft; // forget
        ft.Resize(1, _output_dim);
        Matrix<float> ct; // cell
        ct.Resize(1, _output_dim);
        Matrix<float> ot; // output
        ot.Resize(1, _output_dim);
        for(int t = 0; t < time_steps; ++t){
            // input at time t
            SubMatrix<float> input_t(*in, t, 1, 0, _input_dim);
            SubMatrix<float> output_t_minus_one(*out, (return_sequence ? std::max(t - 1, 0) : 0), 1, 0, _output_dim);
            // input
            AddMatMat(temp_out1, input_t, Wi);
            AddMatMat(temp_out2, output_t_minus_one, Ui);
            AddMat(temp_out1, temp_out2);
            AddBias(temp_out1, Bi);
            ApplySigmoid(it, temp_out1);
            //forget
            AddMatMat(temp_out1, input_t, Wf);
            AddMatMat(temp_out2, output_t_minus_one, Uf);
            AddMat(temp_out1, temp_out2);
            AddBias(temp_out1, Bf);
            ApplySigmoid(ft, temp_out1);
            // cell
            AddMatMat(temp_out1, input_t, Wc);
            AddMatMat(temp_out2, output_t_minus_one, Uc);
            AddMat(temp_out1, temp_out2);
            AddBias(temp_out1, Bc);
            ApplyTanh(ct, temp_out1);
            // update cell state
            MultiplyMat(temp_out1, ft, cell_state);
            MultiplyMat(temp_out2, it, ct);
            AddMat(cell_state, temp_out1, temp_out2); // update cell_state

            //output gate
            AddMatMat(temp_out1, input_t, Wo);
            AddMatMat(temp_out2, output_t_minus_one, Uo);
            AddMat(temp_out1, temp_out2);
            AddBias(temp_out1, Bo);
            ApplySigmoid(ot, temp_out1);
            // final output
            ApplyTanh(temp_out1, cell_state);
            SubMatrix<float> output_t(*out, (return_sequence ? t : 0), 1, 0, _output_dim);
            MultiplyMat(output_t, ot, temp_out1);
        }
    }

}// end of namespace DNN
