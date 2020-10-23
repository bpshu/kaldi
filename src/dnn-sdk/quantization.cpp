

#include "quantization.h"
#include "nnet-math.h"
#include <cassert>
namespace DNN {
        template<typename T>
        Quantizer<T>::Quantizer() : scale_(1.0f), bias_(0.0f){}
        template<typename T>
        Quantizer<T>::Quantizer(float max){
            this->Reset(max);
        }
        template<typename T>
        void Quantizer<T>::Reset(float max){
            return; //reset NULL
        }
        template<typename T>
        Quantizer<T>::~Quantizer() = default;
        template<typename T>
        Quantizer<T>& Quantizer<T>::operator=(const Quantizer& other) = default;
        template<typename T>
        Quantizer<T>::Quantizer(const Quantizer& other) = default;
    
        template<typename T>
        T Quantizer<T>::Quantize(float x) const{
            return SaturateInt<T>(round(x * scale_ + bias_));
        }
        template<typename T>
        float Quantizer<T>::Dequantize(T x) const{
            return (static_cast<float>(x) - bias_) / scale_;
        }
        template<typename T>
        void Quantizer<T>::Quantize(T* dst, unsigned dst_stride, const float* src, unsigned rows, unsigned cols, unsigned src_stride) const{
            Convert(dst, dst_stride, src, rows, cols, src_stride, scale_, bias_);
        }
        template<typename T>
        float Quantizer<T>::Scale() const{
            return scale_;
        }
        template<typename T>
        float Quantizer<T>::Bias() const{
            return bias_;
        }
        // explicit instantiation
        template class Quantizer<char>;
        template class Quantizer<unsigned char>;
        template class Quantizer<float>; // actually it does nothing
}
