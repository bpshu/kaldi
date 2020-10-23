

#ifndef tdnn_h
#define tdnn_h
#include "activation.h"
#include <vector>

namespace DNN{
    class Tdnn : public Layer{
    public:
        Tdnn(int32_t i_dim, int32_t o_dim,
             ACTIVATION_TYPE a_t = ACTIVATION_TYPE::NONE,
             bool norm = true);
        Tdnn(int32_t i_dim, int32_t o_dim,
             const std::vector<int32_t>& input_context,
             ACTIVATION_TYPE a_t = ACTIVATION_TYPE::NONE,
             bool norm = true);
        Tdnn(const Matrix<float>& weights, const Vector<float>& bias,
             const std::vector<int>& input_context,
             ACTIVATION_TYPE a_t = ACTIVATION_TYPE::NONE,
             bool norm_ = true);
        ~Tdnn();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        virtual void ReadData(std::istream &is, bool binary) override;
        void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                          std::vector<int>* target_frames = nullptr) const override;
        void Activate(MatrixBase<float>* in, MatrixBase<float> *out,
                      std::vector<int>* target_frames = nullptr) const;
        void SetActivation(ACTIVATION_TYPE ac);
        const Matrix<float>& GetWeights() const { return weights; }
        const Vector<float>& GetBias() const{ return bias;}
        ACTIVATION_TYPE GetActivationType() const{return ac_type; }

    private:
        Matrix<float> weights;
        Vector<float> bias;
        Activation* activation = nullptr;
        ACTIVATION_TYPE ac_type = ACTIVATION_TYPE::NONE;
        bool norm;

    };
}// end of namespace DNN

#endif /* tdnn_h */
