
#ifndef nnet_math_h
#define nnet_math_h
#include "nnet-matrix.h"
#include "nnet-vector.h"
#include <limits>
#include <climits>
#include <numeric>
#include <cassert>

namespace DNN{
    template<typename T>
    void AddMatMat(MatrixBase<T>& Out, const MatrixBase<T>& X, const MatrixBase<T>& W,
                   std::vector<int>* target_frames = nullptr);
    template<typename T>
    void AddMatMat(MatrixBase<T>& Out, const MatrixBase<T>& X, const MatrixBase<T>& W,
                   const std::vector<int>& context,
                   std::vector<int>* target_frames = nullptr);
    template<typename T>
    void AddMat(MatrixBase<T>& X, const MatrixBase<T>& Temp,
                std::vector<int>* target_frames = nullptr);
    template<typename T>
    void AddMat(MatrixBase<T>& X, const MatrixBase<T>& Temp1, const MatrixBase<T>& Temp2,
                std::vector<int>* target_frames = nullptr);
    template<typename T>
    void AddMatBias(MatrixBase<T>& X, const MatrixBase<T>& Temp, const Vector<T>& Bias,
                std::vector<int>* target_frames = nullptr);
    template<typename T>
    void MultiplyMat(MatrixBase<T>& X, const MatrixBase<T>& Temp,
                     std::vector<int>* target_frames = nullptr);
    template<typename T>
    void MultiplyMat(MatrixBase<T>& X, const MatrixBase<T>& Temp1, const MatrixBase<T>& Temp2,
                     std::vector<int>* target_frames = nullptr);
    template<typename T>
    void AddBias(MatrixBase<T>& X, const Vector<T>& Bias,
                 std::vector<int>* target_frames = nullptr);
    template<typename T>
    void ApplySigmoid(MatrixBase<T>& dst, const MatrixBase<T>& src,
                      std::vector<int>* target_frames = nullptr);
    
    template<typename T>
    void ApplySoftmax(MatrixBase<T>& dst, const MatrixBase<T>& src,
                      std::vector<int>* target_frames = nullptr);
    
    template<typename T>
    void ApplyLogSoftmax(MatrixBase<T>& dst, const MatrixBase<T>& src,
                         std::vector<int>* target_frames = nullptr);
    
    template<typename T>
    void ApplyTanh(MatrixBase<T>& dst, const MatrixBase<T>& src,
                   std::vector<int>* target_frames = nullptr);
    
    template<typename T>
    void ApplyRelu(MatrixBase<T>& dst, const MatrixBase<T>& src,
                   std::vector<int>* target_frames = nullptr);
    
    template<typename T>
    void Normalization(MatrixBase<T>& dst, const MatrixBase<T>& src,
                       std::vector<int>* target_frames = nullptr);
    
    template<typename T>
    void Convert(T* mO, unsigned out_stride, const float* mI, unsigned numRows, unsigned numCols, unsigned in_stride,  float scalingFctr, float biasFctr);
    
    template<typename T>
    inline T SaturateInt(int x){
        if(x > std::numeric_limits<T>::max()){
            x = static_cast<int>(std::numeric_limits<T>::max());
        }
        if(x < std::numeric_limits<T>::min()){
            x = static_cast<int>(std::numeric_limits<T>::min());
        }
        return static_cast<T>(x);
    }
    
    const std::vector<int> Range(int start, int end);
    
    void vector_product(const float * start_a, const float * start_b, float & result,  int cnt);
    float quantized_vector_product(const size_t vectorSize,
                                   const unsigned char *quantizedInput,
                                   const char *weights);
    float max_simd(const float *a, int n);
    float min_simd(const float *a, int n);
    void multiplyMat(float* x, const float* t, int size);
    void multiplyMat(float* x, const float* t1, const float* t2, int size);
    void addMat(float* x, const float* t, int size);
    void addMat(float* x, const float* t1, const float* t2, int size);
    void addMatBias(float* x, const float* t, const float* bias, int size);
    void relu(const float* src, float* dst, int size);
    void vector_scale(const float* src, float* dst, float scale, int size);
    void logMat(float* x, int row, int col, int stride);
} // end of namespace DNN
#include "nnet-math-imp.h"
#endif /* nnet_math_h */
