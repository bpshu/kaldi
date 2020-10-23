
#ifndef bi_rnn_hpp
#define bi_rnn_hpp

#include "activation.h"
namespace DNN{
    class BRNN : public Layer{
    public:
        BRNN(int32_t i_dim, int32_t cell, ACTIVATION_TYPE a_t = ACTIVATION_TYPE::SIGMOID, size_t offset_input = 0);
        ~BRNN();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        virtual void ReadData(std::istream &is, bool binary) override;
        virtual void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                                  std::vector<int>* target_frames = nullptr) const override;
        void SetCoef(const Matrix<float>& w_fwd, const Matrix<float>& u_fwd, const Vector<float>& b1_fwd, const Vector<float>& b2_fwd, const Matrix<float>& w_back, const Matrix<float>& u_back, const Vector<float>& b1_back, const Vector<float>& b2_back);
        ACTIVATION_TYPE GetActivationType() const{return ac_type; }
        void Activate(MatrixBase<float>* in, MatrixBase<float> *out,
                      std::vector<int>* target_frames = nullptr) const;
        void SetActivation(ACTIVATION_TYPE ac);

    private:
        Matrix<float> W_fwd;
        Matrix<float> U_fwd;
        Vector<float> B1_fwd;
        Vector<float> B2_fwd;
        
        Matrix<float> W_back;
        Matrix<float> U_back;
        Vector<float> B1_back;
        Vector<float> B2_back;
        Matrix<float> initial;

        Activation* activation = nullptr;
        ACTIVATION_TYPE ac_type = ACTIVATION_TYPE::NONE;
        size_t offset_input_;
    };
    
}// end of namespace DNN

#endif /* bi_rnn_hpp */
