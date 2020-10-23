

#ifndef rnn_h
#define rnn_h

#include "activation.h"
namespace DNN{
    class RNN : public Layer{
    public:
        RNN(int32_t i_dim, int32_t cell, ACTIVATION_TYPE a_t = ACTIVATION_TYPE::SIGMOID, bool r_seq = false, size_t offset_input = 0);
        ~RNN();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        virtual void ReadData(std::istream &is, bool binary) override;
        bool GetReturnSeq() const;
        void SetReturnSeq(bool ret);
        virtual void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                                  std::vector<int>* target_frames = nullptr) const override;
        void SetCoef(const Matrix<float>& w, const Matrix<float>& u, const Vector<float>& b1, const Vector<float>& b2);
        ACTIVATION_TYPE GetActivationType() const{return ac_type; }
        void Activate(MatrixBase<float>* in, MatrixBase<float> *out,
                      std::vector<int>* target_frames = nullptr) const;
        void SetActivation(ACTIVATION_TYPE ac);

    private:
        bool return_sequence;
        Matrix<float> W;
        Matrix<float> U;
        Vector<float> B1;
        Vector<float> B2;
        Matrix<float> initial;

        Activation* activation = nullptr;
        ACTIVATION_TYPE ac_type = ACTIVATION_TYPE::NONE;
        size_t offset_input_;
    };
    
}// end of namespace DNN

#endif /* rnn_h */
