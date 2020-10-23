

#ifndef activation_h
#define activation_h

#include "layer.h"

namespace DNN{
    enum class ACTIVATION_TYPE{
        SIGMOID,
        RELU,
        TANH,
        SOFTMAX,
        LOG_SOFTMAX,
        NONE
    };

    std::string ACTypeToMarker(ACTIVATION_TYPE type);
    class Activation : public Layer{
    public:
        Activation();
        Activation(int32_t dim);
        ~Activation();
        virtual void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                                  std::vector<int>* target_frames = nullptr) const = 0;

    };
    
    class Sigmoid : public Activation{
    public:
        Sigmoid();
        Sigmoid(int32_t dim);
        ~Sigmoid();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                          std::vector<int>* target_frames = nullptr) const override;

    };
    class Softmax : public Activation{
    public:
        Softmax();
        Softmax(int32_t dim);
        ~Softmax();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                          std::vector<int>* target_frames = nullptr) const override;

    };
    class Tanh : public Activation{
    public:
        Tanh();
        Tanh(int32_t dim);
        ~Tanh();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                          std::vector<int>* target_frames = nullptr) const override;

    };
    class Relu : public Activation{
    public:
        Relu();
        Relu(int32_t dim);
        ~Relu();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                          std::vector<int>* target_frames = nullptr) const override;

    };
    class LogSoftmax : public Activation{
    public:
        LogSoftmax();
        LogSoftmax(int32_t dim);
        ~LogSoftmax();
        virtual void WriteData(std::ostream &os, bool binary) const override;
        void PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                          std::vector<int>* target_frames = nullptr) const override;
    };
    
    inline Activation* CreateActivation(ACTIVATION_TYPE a_t, int32_t dim = 0){
        switch(a_t){
            case ACTIVATION_TYPE::SIGMOID:
                return new Sigmoid(dim);
            case ACTIVATION_TYPE::SOFTMAX:
                return new Softmax(dim);
            case ACTIVATION_TYPE::TANH:
                return new Tanh(dim);
            case ACTIVATION_TYPE::RELU:
                return new Relu(dim);
            case ACTIVATION_TYPE::LOG_SOFTMAX:
                return new LogSoftmax(dim);
            case ACTIVATION_TYPE::NONE:
                return nullptr;
            default:
                return nullptr;
        }
    }
    inline std::string ACTypeToMarker(ACTIVATION_TYPE a_t){
        switch(a_t){
            case ACTIVATION_TYPE::SIGMOID:
                return "SIGMOID";
            case ACTIVATION_TYPE::SOFTMAX:
                return "SOFTMAX";
            case ACTIVATION_TYPE::TANH:
                return "TANH";
            case ACTIVATION_TYPE::RELU:
                return "RELU";
            case ACTIVATION_TYPE::LOG_SOFTMAX:
                return "LOG_SOFMAX";
            case ACTIVATION_TYPE::NONE:
                return "NONE";
            default:
                return "NONE";
        }
    }
    inline ACTIVATION_TYPE MarkerToACtype(const std::string& token){
            if("SIGMOID" == token)
                return ACTIVATION_TYPE::SIGMOID;
            if("SOFTMAX" == token)
                return ACTIVATION_TYPE::SOFTMAX;
            if("TANH" == token)
                return ACTIVATION_TYPE::TANH;
            if("RELU" == token)
                return ACTIVATION_TYPE::RELU;
            if("LOG_SOFTMAX" == token)
                return ACTIVATION_TYPE::LOG_SOFTMAX;
            if("NONE" == token)
                return ACTIVATION_TYPE::NONE;
            return ACTIVATION_TYPE::NONE;
    }
}// end of namespace DNN

#endif /* activation_h */
