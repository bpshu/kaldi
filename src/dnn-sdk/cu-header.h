

#ifndef cu_header_h
#define cu_header_h
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <iostream>


extern const char* cublasGetStatusString(cublasStatus_t status);


#define CU_SAFE_CALL(fun) \
{ \
int32_t ret; \
if ((ret = (fun)) != 0) { \
std::cerr << "cudaError_t " << ret << " : \"" << cudaGetErrorString((cudaError_t)ret) << "\" returned from '" << #fun << "'" << std::endl; exit(1);\
} \
}
#define CUBLAS_SAFE_CALL(fun) \
{ \
int32_t ret; \
if ((ret = (fun)) != 0) { \
std::cerr << "cublasStatus_t " << ret << " : \"" << cublasGetStatusString((cublasStatus_t)ret) << "\" returned from '" << #fun << "'" << std::endl; exit(1); \
} \
}
#endif
#endif /* cu_header_h */
