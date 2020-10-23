

#ifndef kernals_h
#define kernals_h
#include "cu-header.h"
// The size of a CUDA 1-d block, e.g. for vector operations..
#define CU1DBLOCK 256

// The size of edge of CUDA square block, e.g. for matrix operations.
// Must be defined the same in cu-kernels-ansi.h
#define CU2DBLOCK 16
#ifdef USE_CUDA
inline int32_t n_blocks(int32_t size, int32_t block_size) {
    return size / block_size + ((size % block_size == 0)? 0 : 1);
}
void GetBlockSizesForSimpleMatrixOperation(int32_t num_rows,
                                                  int32_t num_cols,
                                                  dim3 *dimGrid,
                                                  dim3 *dimBlock);
//float VectorDotProduct(const float *V1 , const float *V2, int N);

void CudaMatrixMultiplication(const float *A, const float *B, float *C, int m, int n, int k);

void AddVecToRows(const float* row, float* dst, int rows, int cols, int stride);

void Sigmoid(float* dst, const float* src, int rows, int cols, int stride, int src_stride);

void ApplySoftMaxPerRow(float* dst, const float* src, int rows, int cols, int stride, int src_stride);

void ApplyLogSoftMaxPerRow(float* dst, const float* src, int rows, int cols, int stride, int src_stride);

void Tanh(float* dst, const float* src, int rows, int cols, int stride, int src_stride);

void ApplyFloor(float* dst, const float* src, float val, int rows, int cols, int stride, int src_stride);

void NormalizePerRow(float* dst, int dst_stride, const float* src, int rows, int cols, int stride, float target_rms, bool add_log_std_dev);

void ApplyLog(float* x, int rows, int cols, int stride);
#endif
#endif /* kernals_h */
