//
//  kernals.cpp
//  inference_acceleration
//
//  Created by Feng Zhou on 2018/7/19.
//  Copyright © 2018 Feng. All rights reserve
//

#include "kernals.h"
#include <math_constants.h>
#include <iostream>
#include <vector>
#include <cassert>
using std::vector;

void GetBlockSizesForSimpleMatrixOperation(int32_t num_rows,
    int32_t num_cols,
    dim3 *dimGrid,
    dim3 *dimBlock)
{
    assert(num_rows > 0 && num_cols > 0);
    int32_t col_blocksize = 64, row_blocksize = 4;
    while (col_blocksize > 1 && (num_cols + (num_cols / 2) <= col_blocksize || num_rows > 65535 * row_blocksize)) {
        col_blocksize /= 2;
        row_blocksize *= 2;
    }

    dimBlock->x = col_blocksize;
    dimBlock->y = row_blocksize;
    dimBlock->z = 1;
    dimGrid->x = n_blocks(num_cols, col_blocksize);
    dimGrid->y = n_blocks(num_rows, row_blocksize);
    dimGrid->z = 1;
}
__global__ void Vector_Dot_Product (const float *V1 , const float *V2 , float *V3, int N)
{
    __shared__ float chache[CU1DBLOCK] ;
    
    float temp = 0.0;
    
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    
    const unsigned int chacheindex = threadIdx.x ;
    
    while ( tid < N )
    {
        temp += V1[tid] * V2[tid] ;
        
        tid += blockDim.x * gridDim.x ;
    }
    
    chache[chacheindex] = temp ;
    
    __syncthreads () ;
    
    int i  = blockDim.x / 2 ;
    
    while ( i!=0 )
    {
        
        if ( chacheindex < i )
            chache[chacheindex] += chache [chacheindex + i] ;
        
        __syncthreads () ;
        
        i/=2 ;
    }
    
    if ( chacheindex == 0 )
        V3[blockIdx.x] = chache [0] ;
}

__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

__global__ void gpu_matrix_mult(const float *a, const float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < k && row < m)
    {
        float sum = 0.0;
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[col * n + i];
        }
        c[row * k + col] = sum;
    }
}
__global__ void gpu_add_vec_to_rows(const float* row, float* dst, int rows, int cols, int stride)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t index = i + j * stride;
    if (i < cols && j < rows) dst[index] = row[i] + dst[index];
}

void CudaMatrixMultiplication(const float *A, const float *B, float *C, int m, int n, int k){
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(k, CU2DBLOCK),
                 n_blocks(m, CU2DBLOCK));
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(A, B, C, m, n, k);
}
__global__ void gpu_sigmoid(float* y, const float* x, int rows, int cols, int stride, int src_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_index = i + j * stride, src_index = i + j * src_stride;
    if (i < cols && j < rows) {
        float res = 1.0 / (1.0 + exp(-x[src_index]));
        y[dst_index] = res;
    }
}
__global__ static void gpu_floor(float* mat, const float* src, float floor_val, int rows, int cols, int stride, int src_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    int index = i + j * stride, src_index = i + j * src_stride;

    if (i < cols && j < rows) {
        mat[index] = fmax(src[src_index], floor_val);
    }
}
__global__ void gpu_apply_log(float* mat, int rows, int cols, int stride){
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    int index = i + j * stride;
    if(i < cols && j < rows)
	mat[index] = log(mat[index] + 1e-20);

}
__global__ void gpu_tanh(float* y, const float* x, int rows, int cols, int stride, int src_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_index = i + j * stride, src_index = i + j * src_stride;
    if (i < cols && j < rows) {
        float exp_2x = exp(2.0 * x[src_index]);
        float res;
        if (isinf(exp_2x)) {
            res = 1.0;
        } else {
            res = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
        y[dst_index] = res;
    }
}

__global__ void gpu_softmax(float* y, const float* x, int rows, int cols, int stride, int src_stride)
{
    __shared__ float smem[CU1DBLOCK];
    const int i = blockIdx.x;
    const int x_start = i * src_stride;
    const int y_start = i * stride;
    const int tid = threadIdx.x;

    float tmax = sizeof(float) == sizeof(float) ? -CUDART_INF_F : -CUDART_INF;
    for (int j = tid; j < cols; j += CU1DBLOCK) {
    tmax = fmax(tmax, x[x_start + j]);
    }
    smem[tid] = tmax;
    __syncthreads();
# pragma unroll
    for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
        if (tid < shift) {
            smem[tid] = fmax(smem[tid], smem[tid + shift]);
        }
        __syncthreads();
    }
    if (tid < warpSize) {
#   pragma unroll
        for (int shift = warpSize; shift > 0; shift >>= 1) {
            smem[tid] = fmax(smem[tid], smem[tid + shift]);
        }
    }

    __syncthreads();
    float max = smem[0];

    float tsum = float(0);
    for (int j = tid; j < cols; j += CU1DBLOCK) {
        tsum += exp(x[x_start + j] - max);
    }
    smem[tid] = tsum;
    __syncthreads();

# pragma unroll
    for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
        if (tid < shift) {
            smem[tid] += smem[tid + shift];
        }
        __syncthreads();
    }

    if (tid < warpSize) {
#   pragma unroll
        for (int shift = warpSize; shift > 0; shift >>= 1) {
            smem[tid] += smem[tid + shift];
        }
    }

    __syncthreads();
    float inv_sum = float(1) / smem[0];

    for (int j = tid; j < cols; j += CU1DBLOCK) {
        y[y_start + j] = exp(x[x_start + j] - max) * inv_sum;
    }

}
__global__ void gpu_logsoftmax(float* y, const float* x, int rows, int cols, int stride, int x_stride){
    __shared__ float smem[CU1DBLOCK];
    const int i = blockIdx.x;
    const int x_start = i * x_stride;
    const int y_start = i * stride;
    const int tid = threadIdx.x;

    // find max element of the row
    // reduce to CU1DBLOCK elements per row.
    float tmax = -1e20;
    for (int j = tid; j < cols; j += CU1DBLOCK) {
        tmax = fmax(tmax, x[x_start + j]);
    }
    smem[tid] = tmax;
    __syncthreads();

    // reduce to 2x warpSize elements per row
# pragma unroll
    for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
        if (tid < shift) {
            smem[tid] = fmax(smem[tid], smem[tid + shift]);
        }
        __syncthreads();
    }

    // reduce to 1 element per row
    if (tid < warpSize) {
        for (int shift = warpSize; shift > 0; shift >>= 1) {
            smem[tid] = fmax(smem[tid], smem[tid + shift]);
        }
    }

    // broadcast max to all threads
    __syncthreads();
    float max = smem[0];

    // sum_j(exp(x(i,j)-max))
    // reduce to CU1DBLOCK elements per row.
    float tsum = float(0);
    for (int j = tid; j < cols; j += CU1DBLOCK) {
        tsum += exp(x[x_start + j] - max);
    }
    smem[tid] = tsum;
    __syncthreads();

    // reduce to 2x warpSize elements per row
# pragma unroll
    for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
        if (tid < shift) {
            smem[tid] += smem[tid + shift];
        }
        __syncthreads();
    }

    // reduce to 1 element per row
    if (tid < warpSize) {
        for (int shift = warpSize; shift > 0; shift >>= 1) {
            smem[tid] += smem[tid + shift];
        }
    }

    // broadcast sum to all threads
    __syncthreads();
    float log_sum = log(smem[0]);

    // normalize the row
    for (int j = tid; j < cols; j += CU1DBLOCK) {
        y[y_start + j] = x[x_start + j] - max - log_sum;
    }
}

__global__ static void gpu_normalize_per_row(float *y, int y_stride, const float *x, int rows, int cols, int stride, float target_rms, bool add_log_stddev) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + i * stride;
    __shared__ float ssum[CU1DBLOCK];

    // Reduce x_j^2 to CU1DBLOCK elements per row
    float tsum = float(0);
    for (int j = tid; j < cols; j += CU1DBLOCK) {
        tsum += x_row[j] * x_row[j];
    }
    ssum[tid] = tsum;
    __syncthreads();

    // Tree reduce to 2x warpSize elements per row
# pragma unroll
    for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
        if (tid < shift)
            ssum[tid] += ssum[tid + shift];
        __syncthreads();
    }

    // Reduce last warp to 1 element per row.
    // Threads implicitly synchronized within a warp.
    if (tid < warpSize) {
#   pragma unroll
        for (int shift = warpSize; shift > 0; shift >>= 1) {
            ssum[tid] += ssum[tid + shift];
        }
    }

    const float kSquaredNormFloor = 1.3552527156068805425e-20; // 2^-66
    if (tid == 0) {
        ssum[0] = sqrt(
        fmax(ssum[0] / (target_rms * target_rms * cols), kSquaredNormFloor));
    }

    // Broadcast floored stddev to all threads.
    __syncthreads();
    const float stddev_div_target_rms = ssum[0];
    const float scale = float(1) / stddev_div_target_rms;

    // Store normalized input to output
    float* y_row = y + i * y_stride;
    for (int j = tid; j < cols; j += CU1DBLOCK) {
        y_row[j] = x_row[j] * scale;
    }

    if (tid == 0 && add_log_stddev) {
        y_row[cols] = log(stddev_div_target_rms * target_rms);
    }
}


void AddVecToRows(const float* row, float* dst, int rows, int cols, int stride){
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(rows, cols, &dimGrid, &dimBlock);
    gpu_add_vec_to_rows<<<dimGrid, dimBlock>>>(row, dst, rows, cols, stride);
}

void Sigmoid(float* dst, const float* src, int rows, int cols, int stride, int src_stride){
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(rows, cols, &dimGrid, &dimBlock);
    gpu_sigmoid<<<dimGrid, dimBlock>>>(dst, src, rows, cols, stride, src_stride);
}
void ApplySoftMaxPerRow(float* dst, const float* src, int rows, int cols, int stride, int src_stride){
    size_t dimBlock = CU1DBLOCK;
    size_t dimGrid = rows;
    gpu_softmax<<<dimGrid, dimBlock>>>(dst, src, rows, cols, stride, src_stride);
}
void ApplyLogSoftMaxPerRow(float* dst, const float* src, int rows, int cols, int stride, int src_stride){
    size_t dimBlock = CU1DBLOCK;
    size_t dimGrid = rows;
    gpu_logsoftmax<<<dimGrid, dimBlock>>>(dst, src, rows, cols, stride, src_stride);
}
void Tanh(float* dst, const float* src, int rows, int cols, int stride, int src_stride){
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(rows, cols, &dimGrid, &dimBlock);
    gpu_tanh<<<dimGrid, dimBlock>>>(dst, src, rows, cols, stride, src_stride);
}
void ApplyFloor(float* dst, const float* src, float val, int rows, int cols, int stride, int src_stride){
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(rows, cols, &dimGrid, &dimBlock);
    gpu_floor<<<dimGrid, dimBlock>>>(dst, src, val, rows, cols, stride, src_stride);
}
void NormalizePerRow(float* dst, int dst_stride, const float* src, int rows, int cols, int stride, float target_rms, bool add_log_std_dev){
    size_t dimBlock = CU1DBLOCK;
    size_t dimGrid = rows;
    gpu_normalize_per_row<<<dimGrid, dimBlock>>>(dst, dst_stride, src, rows, cols, stride, target_rms, add_log_std_dev);
}
void ApplyLog(float* x, int rows, int cols, int stride){
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(rows, cols, &dimGrid, &dimBlock);
    gpu_apply_log<<<dimGrid, dimBlock>>>(x, rows, cols, stride);
}
