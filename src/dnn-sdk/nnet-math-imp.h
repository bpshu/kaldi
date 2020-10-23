
#ifndef nnet_math_imp_h
#define nnet_math_imp_h

#ifdef USE_CUDA
#include "kernals.h"
#endif

#include <vector>
#include <fstream>

using std::vector;

namespace DNN{
    template<>
    inline void AddMatMat<float>(MatrixBase<float>& Out, const MatrixBase<float>& X, const MatrixBase<float>& W,
                                 std::vector<int>* target_frames){
        // for example
        // X is the input, while W is the Weight, actually it is doing
        // matrix multipication of X * W.transpose, so
        LOG_ASSERT(X.NumCols() == W.NumCols());
        LOG_ASSERT(Out.NumRows() == X.NumRows());
        LOG_ASSERT(Out.NumCols() == W.NumRows());
#ifdef USE_CUDA
        auto m =  W.NumRows();
        auto n =  X.NumRows();
        auto k =  W.NumCols();
        auto k1 = X.NumCols();
        LOG_ASSERT(k == k1);
        constexpr float alph = 1.0f;
        constexpr float beta = 0.0f;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alph, W.Data(), W.Stride(), X.Data(), X.Stride(), &beta, Out.Data(), Out.Stride());
        cublasDestroy(handle);
//        CudaMatrixBaseMultiplication(X.Data(), W.Data(), Out.Data(), X.NumRows(), X.Stride(), W.NumRows());
#else
        for(int rX : (target_frames ? *target_frames : Range(0, X.NumRows()))){
            if(rX % MatrixBase<float>::Skip() != 0){
                memcpy((void *)Out.RowData(rX), (void *)Out.RowData(rX - rX % MatrixBase<float>::Skip()), Out.Stride()*sizeof(float));
                continue;
            }
            const float* x = X.RowData(rX);
            for(int rW = 0; rW < W.NumRows(); ++rW){
                const float* w = W.RowData(rW);
                float rst = 0.0;
                vector_product(x, w, rst, X.NumCols());
                Out(rX, rW) = rst;
            }
        }
#endif
    }
    template<>
    inline void AddMatMat<float>(MatrixBase<float>& Out, const MatrixBase<float>& X, const MatrixBase<float>& W,
                                 const std::vector<int>& context,
                                 std::vector<int>* target_frames){
        // for example
        // X is the input, while W is the Weight, actually it is doing
        // matrix multipication of X * W.transpose, so
        LOG_ASSERT(!context.empty());
        int low = context.front();
        int high = context.back();
        int diff = high - low;
        LOG_ASSERT(diff >= 0); // context should already be sorted!
        LOG_ASSERT(X.NumCols() * context.size() == W.NumCols());
        LOG_ASSERT(Out.NumRows() == X.NumRows() - diff); // frames
        LOG_ASSERT(Out.NumCols() == W.NumRows());
#ifdef USE_CUDA
        // we will make a new X
        Matrix<float> X_new;
        X_new.Resize(Out.NumRows(), W.NumCols());

        auto dst = X_new.Data();
        for(int j = 0; j < context.size(); ++j){
	    int row_num = (j == 0? std::max(low, 0) : context[j] - low);
	    LOG_ASSERT(row_num <= diff);
            CU_SAFE_CALL(cudaMemcpy2D(dst, X_new.Stride()*sizeof(float), X.RowData(row_num), X.Stride()*sizeof(float), X.NumCols()*sizeof(float), Out.NumRows(),cudaMemcpyDeviceToDevice));
            dst += X.NumCols();
        }
        constexpr float alph = 1.0f;
        constexpr float beta = 0.0f;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm_v2(/*GetCuHandler()*/handle, CUBLAS_OP_T, CUBLAS_OP_N, W.NumRows(), X_new.NumRows(), W.NumCols(), &alph, W.Data(), W.Stride(), X_new.Data(), X_new.Stride(), &beta, Out.Data(), Out.Stride());
        cublasDestroy(handle);
#else
        float *x = AlignedAlloc<float>(W.Stride());
        memset((void*)x, 0, W.Stride()*sizeof(float));
        if(target_frames == nullptr){
            for(int rX = 0; rX < X.NumRows(); ++rX){
                if(rX + low < 0 || rX + low >= X.NumRows()
                   || rX + high < 0 || rX + high >= X.NumRows()){
                    continue;
                }
                if((rX + low) % MatrixBase<float>::Skip() != 0){
                    memcpy((void *)Out.RowData(rX + low), (void *)Out.RowData((rX + low) - (rX + low) % MatrixBase<float>::Skip()), Out.Stride()*sizeof(float));
                    continue;
                }
                for(int j = 0; j < context.size(); ++j){
                    int ind = rX + context[j];
                    LOG_ASSERT(ind >= 0 && ind < X.NumRows());
                    memcpy((void*)(x + j * X.NumCols()), (void*)X.RowData(ind), X.NumCols()*sizeof(float));
                }
                // now we got appended x by context
                for(int rW = 0; rW < W.NumRows(); ++rW){
                    const float* w = W.RowData(rW);
                    float rst = 0.0;
                    vector_product(x, w, rst, W.NumCols());
                    Out(rX + low, rW) = rst;
                }
            }
        }else{
            for(int rX : *target_frames){
                for(int j = 0; j < context.size(); ++j){
                    int ind = rX - low + context[j];
                    LOG_ASSERT(ind >= 0 && ind < X.NumRows());
                    memcpy((void*)(x + j * X.NumCols()), (void*)X.RowData(ind), X.NumCols()*sizeof(float));
                }
                // now we got appended x by context
                for(int rW = 0; rW < W.NumRows(); ++rW){
                    const float* w = W.RowData(rW);
                    float rst = 0.0;
                    vector_product(x, w, rst, W.NumCols());
                    Out(rX, rW) = rst;
                }
            }
        }
        aligned_free(x);
#endif
    }

    template<>
    inline void MultiplyMat<float>(MatrixBase<float>& X, const MatrixBase<float>& Temp,
                                   std::vector<int>* target_frames){
        LOG_ASSERT(X.NumCols() == Temp.NumCols());
        LOG_ASSERT(X.NumRows() == Temp.NumRows());
#ifdef USE_CUDA
        LOG_ASSERT(0 && "MultiplyMat not supported in GPU");
#else
        for(int r : (target_frames ? *target_frames : Range(0, X.NumRows()))){
            float* x = X.RowData(r);
            const float* t = Temp.RowData(r);
            multiplyMat(x, t, X.Stride());
        }
#endif
    }
    template<>
    inline void MultiplyMat<float>(MatrixBase<float>& X, const MatrixBase<float>& Temp1, const MatrixBase<float>& Temp2, std::vector<int>* target_frames){
        LOG_ASSERT(X.NumCols() == Temp1.NumCols() && Temp1.NumCols() == Temp2.NumCols());
        LOG_ASSERT(X.NumRows() == Temp1.NumRows() && Temp1.NumRows() == Temp2.NumRows());
#ifdef USE_CUDA
        LOG_ASSERT(0 && "MultiplyMat not supported in GPU");
#else
        for(int r : (target_frames ? *target_frames : Range(0, X.NumRows()))){
            float* x = X.RowData(r);
            const float* t1 = Temp1.RowData(r);
            const float* t2 = Temp2.RowData(r);
            multiplyMat(x, t1, t2, X.Stride());
        }
#endif
    }
    template<>
    inline void AddMat<float>(MatrixBase<float>& X, const MatrixBase<float>& Temp,
                              std::vector<int>* target_frames){
        LOG_ASSERT(X.NumCols() == Temp.NumCols());
        LOG_ASSERT(X.NumRows() == Temp.NumRows());
#ifdef USE_CUDA
        LOG_ASSERT(0 && "AddMat not supported in GPU");
#else
        for(int r : (target_frames ? *target_frames : Range(0, X.NumRows()))){
            float* x = X.RowData(r);
            const float* t = Temp.RowData(r);
            addMat(x, t, X.Stride());
        }
#endif
    }
    template<>
    inline void AddMat<float>(MatrixBase<float>& X, const MatrixBase<float>& Temp1,
                              const MatrixBase<float>& Temp2, std::vector<int>* target_frames){
        LOG_ASSERT(X.NumCols() == Temp1.NumCols() && Temp1.NumCols() == Temp2.NumCols());
        LOG_ASSERT(X.NumRows() == Temp1.NumRows() && Temp1.NumRows() == Temp2.NumRows());
#ifdef USE_CUDA
        LOG_ASSERT(0 && "AddMat not supported in GPU");
#else
        for(int r : (target_frames ? *target_frames : Range(0, X.NumRows()))){
            float* x = X.RowData(r);
            const float* t1 = Temp1.RowData(r);
            const float* t2 = Temp2.RowData(r);
            addMat(x, t1, t2, X.Stride());
        }
#endif
    }
    template<>
    inline void AddBias<float>(MatrixBase<float>& X, const Vector<float>& Bias,
                               std::vector<int>* target_frames){
        LOG_ASSERT(X.NumCols() == Bias.Dim());
#ifdef USE_CUDA
        AddVecToRows(Bias.Data(), X.Data(), X.NumRows(), X.NumCols(), X.Stride());
#else
        const float* b = Bias.Data();
        for(int r : (target_frames ? *target_frames : Range(0, X.NumRows()))){
            float* x = X.RowData(r);
            addMat(x, b, X.Stride());
        }
#endif
    }
    template<>
    inline void AddMatBias<float>(MatrixBase<float>& X,
                                  const MatrixBase<float>& Temp,
                                  const Vector<float>& Bias,
                                  std::vector<int>* target_frames){
        LOG_ASSERT(X.NumCols() == Bias.Dim());
        LOG_ASSERT(X.NumCols() == Temp.NumCols());
#ifdef USE_CUDA
         LOG_ASSERT(0 && "AddMatBias not supported in GPU");
#else
        const float* b = Bias.Data();
        for(int r : (target_frames ? *target_frames : Range(0, X.NumRows()))){
            float* x = X.RowData(r);
            const float* t = Temp.RowData(r);
            addMatBias(x, t, b, X.Stride());
        }
#endif
    }
    template<>
    inline void ApplySigmoid<float>(MatrixBase<float>& dst, const MatrixBase<float>& src,
                                    std::vector<int>* target_frames){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
        //LOG_DEBUG("test float sigmoid");
#ifdef USE_CUDA
        Sigmoid(dst.Data(), src.Data(), dst.NumRows(), dst.NumCols(), dst.Stride(), src.Stride());
#else
        for(int i : (target_frames ? *target_frames : Range(0, src.NumRows()))){
            const float * from = src.RowData(i);
            float * to = dst.RowData(i);
            for(int j = 0 ;j < src.NumCols(); ++j){
                //hard sigmoid
//                if(from[j] < - 2.5) to[j] = 0;
//                else if(from[j] > 2.5) to[j] = 1;
//                else to[j] =  0.2 * from[j] + 0.5;
                // normal sogmoid
                to[j] = 1.0f / (1.0f + exp( 0 - from[j]));
            }
        }
#endif
    }
    inline void ApplySigmoid(MatrixBase<unsigned char>& dst, const MatrixBase<float>& src,
                             std::vector<int>* target_frames = nullptr){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
        Matrix<float> temp_out;
        temp_out.Resize(dst.NumRows(), dst.NumCols());
        ApplySigmoid(temp_out, src, target_frames);
        Quantizer<unsigned char> quantizer(1.0f);
        LOG_ASSERT(quantizer.Bias() == 0.0f);
        quantizer.Quantize(dst.Data(), dst.Stride(), temp_out.Data(), temp_out.NumRows(), temp_out.NumCols(), temp_out.Stride());
        dst.SetQuantizer(quantizer);
    }
    template<>
    inline void ApplySoftmax<float>(MatrixBase<float>& dst, const MatrixBase<float>& src,
                                    std::vector<int>* target_frames){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
#ifdef USE_CUDA
        ApplySoftMaxPerRow(dst.Data(), src.Data(), dst.NumRows(), dst.NumCols(), dst.Stride(), src.Stride());
#else
        for(int i : (target_frames ? *target_frames : Range(0, src.NumRows()))){
            float sum = 0.0F;
            for(int c = 0; c < src.NumCols(); ++c){
                sum += dst(i,c) = exp(src(i,c));
            }
            for(int c = 0; c < src.NumCols(); ++c){
                dst(i, c) /= sum;
            }
        }
#endif
    }
    inline void ApplySoftmax(MatrixBase<unsigned char>& dst, const MatrixBase<float>& src,
                                    std::vector<int>* target_frames = nullptr){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
        Matrix<float> temp_out;
        temp_out.Resize(dst.NumRows(), dst.NumCols());
        ApplySoftmax(temp_out, src, target_frames);
        Quantizer<unsigned char> quantizer(1.0f);
        LOG_ASSERT(quantizer.Bias() == 0.0f);
        quantizer.Quantize(dst.Data(), dst.Stride(), temp_out.Data(), temp_out.NumRows(), temp_out.NumCols(), temp_out.Stride());
        dst.SetQuantizer(quantizer);
        
    }
    template<>
    inline void ApplyLogSoftmax<float>(MatrixBase<float>& dst, const MatrixBase<float>& src,
                                       std::vector<int>* target_frames){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
#ifdef USE_CUDA
        ApplyLogSoftMaxPerRow(dst.Data(), src.Data(), dst.NumRows(), dst.NumCols(), dst.Stride(), src.Stride());
#else
        for(int i : (target_frames ? *target_frames : Range(0, src.NumRows()))){
            double sum = 0.0;
            for(int c = 0; c < src.NumCols(); ++c){
                sum += exp(static_cast<double>(src(i, c)));
            }
            float s = log(sum);
            for(int c = 0; c < src.NumCols(); ++c){
                dst(i, c) = src(i, c) - s;
            }

        }
#endif
    }
    template<>
    inline void ApplyTanh<float>(MatrixBase<float>& dst, const MatrixBase<float>& src,
                                 std::vector<int>* target_frames){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
#ifdef USE_CUDA
        Tanh(dst.Data(), src.Data(), dst.NumRows(), dst.NumCols(), dst.Stride(), src.Stride());
#else
        float * to = NULL;
        for(int i : (target_frames ? *target_frames : Range(0, src.NumRows()))){
            const float* from = src.RowData(i);
            to = dst.RowData(i);
            for(int j = 0 ;j < src.NumCols(); ++j){
                to[j] = (exp(from[j]) - exp(-from[j]))/(exp(from[j]) + exp(-from[j]));
            }
        }
#endif
    }
    inline void ApplyTanh(MatrixBase<char>& dst, const MatrixBase<float>& src,
                              std::vector<int>* target_frames = nullptr){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
        Matrix<float> temp_out;
        temp_out.Resize(dst.NumRows(), dst.NumCols());
        ApplyTanh(temp_out, src, target_frames);
        Quantizer<char> quantizer(1.0f);
        LOG_ASSERT(quantizer.Bias() == 0.0f);
        quantizer.Quantize(dst.Data(), dst.Stride(), temp_out.Data(), temp_out.NumRows(), temp_out.NumCols(), temp_out.Stride());
        dst.SetQuantizer(quantizer);
    }
    template<>
    inline void ApplyRelu<float>(MatrixBase<float>& dst, const MatrixBase<float>& src,
                                 std::vector<int>* target_frames){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols());
#ifdef USE_CUDA
        ApplyFloor(dst.Data(), src.Data(), 0.0f, dst.NumRows(), dst.NumCols(), dst.Stride(), src.Stride());
#else
        for(int i : (target_frames ? *target_frames : Range(0, src.NumRows()))){
            const float* from = src.RowData(i);
            float * to = dst.RowData(i);
            relu(from, to, src.Stride());
        }
#endif
    }
    template<>
    inline void Normalization<float>(MatrixBase<float>& dst, const MatrixBase<float>& src,
                                     std::vector<int>* target_frames){
        LOG_ASSERT(src.NumRows() == dst.NumRows() && src.NumCols() == dst.NumCols() );
        constexpr float target_rms = 1.0;
#ifdef USE_CUDA
        constexpr bool add_log_std_dev = false;
        NormalizePerRow(dst.Data(), dst.Stride(), src.Data(), src.NumRows(), src.NumCols(), src.Stride(), target_rms, add_log_std_dev);
#else
        
        float dim = float(src.NumCols());
        float gconst = target_rms * sqrt(dim);
        for(int i : (target_frames ? *target_frames : Range(0, src.NumRows()))){
            const float* from = src.RowData(i);
            float rst = 0; // square sum
            vector_product(from, from, rst, src.NumCols());
            float squre_root = sqrt(rst) + 2e-66;// avoid divide by 0
            float scale = gconst / squre_root;
            float * to = dst.RowData(i);
            vector_scale(from, to, scale, dst.Stride());
        }
#endif
    }

}
#endif /* nnet_math_imp_h */
