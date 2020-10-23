

#include <stdio.h>
#include "activation.h"
namespace DNN{
    Activation::Activation():Layer(){}
    Activation::Activation(int32_t dim):Layer(dim, dim){}
    Activation::~Activation(){}
    Sigmoid::Sigmoid(): Activation(){}
    
    Sigmoid::Sigmoid(int32_t dim): Activation(dim){}
    void Sigmoid::WriteData(std::ostream &os, bool binary) const{
        WriteToken(os, binary, "SIGMOID");
    }
    Sigmoid::~Sigmoid(){}

    void Sigmoid::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                               std::vector<int>* target_frames) const{
        ApplySigmoid(*out, *in, target_frames);
    }

    Softmax::Softmax(): Activation(){}
    Softmax::Softmax(int32_t dim): Activation(dim){}
    Softmax::~Softmax(){}
    void Softmax::WriteData(std::ostream &os, bool binary) const{
        WriteToken(os, binary, "SOFTMAX");
    }
    void Softmax::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                               std::vector<int>* target_frames) const{
        ApplySoftmax(*out, *in, target_frames);
    }

    Tanh::Tanh(): Activation(){}
    Tanh::Tanh(int32_t dim): Activation(dim){}
    Tanh::~Tanh(){}
    void Tanh::WriteData(std::ostream &os, bool binary) const{
        WriteToken(os, binary, "TANH");
    }
    void Tanh::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                            std::vector<int>* target_frames) const{
        ApplyTanh(*out, *in, target_frames);
    }

    Relu::Relu(): Activation(){}
    Relu::Relu(int32_t dim): Activation(dim){}
    Relu::~Relu(){}
    void Relu::WriteData(std::ostream &os, bool binary) const{
        WriteToken(os, binary, "RELU");
    }
    void Relu::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                            std::vector<int>* target_frames) const{
        ApplyRelu(*out, *in, target_frames);
    }

    LogSoftmax::LogSoftmax(): Activation(){}
    LogSoftmax::LogSoftmax(int32_t dim): Activation(dim){}
    LogSoftmax::~LogSoftmax(){}
    void LogSoftmax::WriteData(std::ostream &os, bool binary) const{
        WriteToken(os, binary, "LOG_SOFTMAX");
    }
    void LogSoftmax::PropagateFnc(MatrixBase<float>* in, MatrixBase<float> *out,
                                  std::vector<int>* target_frames)const{
        ApplyLogSoftmax(*out, *in, target_frames);
    }
    
}// end of namespace DNN
