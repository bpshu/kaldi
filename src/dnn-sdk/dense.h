

#ifndef dense_h
#define dense_h
#include "activation.h"
namespace DNN{
    // Dense or we call Fully Connected layer
    // Namely, a linear operation on the layer's input vector( W * X + b).
    class Dense : public Layer{
    public:
        Dense(ACTIVATION_TYPE a_t = ACTIVATION_TYPE::NONE);
        Dense(int32_t i_dim, int32_t o_dim, ACTIVATION_TYPE a_t = ACTIVATION_TYPE::NONE);
        Dense(const Matrix<float>& W, const Vector<float>& b, ACTIVATION_TYPE a_t = ACTIVATION_TYPE::NONE);
        ~Dense();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        virtual void ReadData(std::istream &is, bool binary) override;
        void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                          std::vector<int>* target_frames = nullptr) const override;
        void Activate(MatrixBase<float>* in, MatrixBase<float> *out,
                      std::vector<int>* target_frames = nullptr) const ;
        void SetActivation(ACTIVATION_TYPE ac);
        const Matrix<float>& GetWeights() const { return weights; }
        const Vector<float>& GetBias() const{ return bias; }
        ACTIVATION_TYPE GetActivationType() const{return ac_type; }

    private:
        Matrix<float> weights;
        Vector<float> bias;
        Activation* activation = nullptr;
        ACTIVATION_TYPE ac_type = ACTIVATION_TYPE::NONE;

    };
}// end of namespace DNN

#endif /* dense_h */
