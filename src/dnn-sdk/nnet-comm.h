

#ifndef nnet_comm_h
#define nnet_comm_h

#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <memory>
#ifdef USE_CUDA
#include "cu-header.h"
#endif
#include "quantization.h"
namespace DNN{
    #define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)
    constexpr float SIGMOID_QUANTIZATION_MULTIPLIER = 255.0f;
    constexpr unsigned char SIGMOID_QUANTIZATION_MULTIPLIER_UCHAR = 255;
    constexpr int SIGMOID_LOOKUP_SIZE = 1280; // arbitrary 64*28
    constexpr int SIGMOID_HALF_LOOKUP_SIZE = SIGMOID_LOOKUP_SIZE / 2;
    constexpr float WEIGHT_MULTIPLIER = 127;
    
    constexpr float NORM_QUANTIZATION_MULTIPLIER = 255.0f;
    constexpr float NORM_CUTOFF = 20.0f;
    constexpr float RELU_QUANTIZATION_MULTIPLIER = 255.0f;
    constexpr float CHAR_MULTIPLIER = 127.0f;

#ifdef USE_SSE
    constexpr int ALIGN = 16;
#elif defined USE_AVX2
    constexpr int ALIGN = 32;
#elif defined USE_AVX512
    constexpr int ALIGN = 64;
#else
    constexpr int ALIGN = 16;
#endif
    

    
#ifdef DEBUG
#define LOG_DEBUG(X) do { std::cout<<__FILE__<<":"<<__LINE__<<":"<< X<< std::endl;  } while(0)
#else
#define LOG_DEBUG(X) do {} while(0)
#endif
    
#define LOG_ERROR(X) \
    do { std::cerr<<__FILE__<<":"<<__LINE__<<":"<< X << std::endl; exit(-1);} while(0)
    
#define LOG_ASSERT(X) \
    {if(!(X)){std::cerr<<"EXIT "<<__FILE__<<":"<<__LINE__<<":"<<#X<<std::endl; assert(0);}}
#define LOG_WARN(X) \
    {std::cerr <<"("<< __FILE__ <<":" << __LINE__ << ") " << X << std::endl;}
    
    inline void *aligned_malloc(size_t align, size_t size) {
        void *result;
#ifdef _MSC_VER
        result = _aligned_malloc(size, align);
#else
        if (posix_memalign(&result, align, size)) result = nullptr;
#endif
	if(result) LOG_ASSERT(is_aligned(result, align));
        return result;
    }
    
    inline void aligned_free(void *ptr) {
        
#ifdef _MSC_VER
        _aligned_free(ptr);
#elif defined(USE_CUDA)
        CU_SAFE_CALL(cudaFree(ptr));
#else
        free(ptr);
#endif
    }
    template<typename T>
    inline T *AlignedAlloc(size_t count) {
#if defined(USE_SSE) || defined(USE_AVX2) || defined(USE_AVX512)
        return reinterpret_cast<T *> (aligned_malloc(ALIGN, sizeof(T) * count)); 
#else
        return reinterpret_cast<T *> (malloc(sizeof(T) * count));
#endif
    }
    
    
} // end of namespace DNN

#endif /* nnet_comm_h */
