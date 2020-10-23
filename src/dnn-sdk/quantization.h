

#ifndef quantization_h
#define quantization_h

namespace DNN {
    template<typename T>
    class Quantizer{
    public:
        Quantizer();
        Quantizer(float max);
        void Reset(float max);
        ~Quantizer();
        Quantizer& operator=(const Quantizer& other);
        Quantizer(const Quantizer& other);
        
        float Scale() const;
        float Bias() const;
        
        void Quantize(T* dst, unsigned dst_stride, const float* src, unsigned rows, unsigned cols, unsigned src_stride) const;
        T Quantize(float x) const;
        float Dequantize(T x) const;
        
    private:
        float scale_ = 1.0f;
        float bias_ = 0.0f;
    };
}
#endif /* quantization_h */
