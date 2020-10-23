
#ifndef lstm_h
#define lstm_h
#include "activation.h"
namespace DNN{
    class LSTM : public Layer{
    public:
        LSTM(int32_t i_dim, int32_t cell,  bool r_seq = false);
        ~LSTM();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        virtual void ReadData(std::istream &is, bool binary) override;
        bool GetReturnSeq() const;
        void SetReturnSeq(bool ret);
        virtual void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                                  std::vector<int>* target_frames = nullptr) const override;
        void SetCoef(const Matrix<float>& wi, const Matrix<float>& ui, const Vector<float>& bi,
                     const Matrix<float>& wf, const Matrix<float>& uf, const Vector<float>& bf,
                     const Matrix<float>& wc, const Matrix<float>& uc, const Vector<float>& bc,
                     const Matrix<float>& wo, const Matrix<float>& uo, const Vector<float>& bo);

    private:
        bool return_sequence;
        // input gate
        Matrix<float> Wi;
        Matrix<float> Ui;
        Vector<float> Bi;
        // forget gate
        Matrix<float> Wf;
        Matrix<float> Uf;
        Vector<float> Bf;
        // Cell
        Matrix<float> Wc;
        Matrix<float> Uc;
        Vector<float> Bc;
        // output gate
        Matrix<float> Wo;
        Matrix<float> Uo;
        Vector<float> Bo;
        

        
    };
    
    
}// end of namespace DNN


#endif /* lstm_h */
