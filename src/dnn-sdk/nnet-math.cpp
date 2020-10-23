

#include "nnet-math.h"
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "kernals.h"
/*
#ifdef USE_SSE
#include "sse_mathfun.h"
#elif defined USE_AVX2
#include "avx_mathfun.h"
#elif defined USE_AVX512
#include "avx_mathfun.h"
#endif
*/
namespace DNN{
    
    const std::vector<int> Range(int start, int end){
        std::vector<int> ans(end);
        std::iota(ans.begin(), ans.end(), start);
        return ans;
    }
    
#ifdef USE_SSE
    constexpr int SSE_STRIDE = 16;
#elif defined USE_AVX2
    constexpr int AVX2_STRIDE = 32;
#elif defined USE_AVX512
    constexpr int AVX512_STRIDE = 64;
#endif

#ifdef USE_AVX2
    float Avx2IntHorizontalSum(__m256i a) {
        __m256i t1 = _mm256_hadd_epi32(a,a);
        __m256i t2 = _mm256_hadd_epi32(t1,t1);
        __m128i t3 = _mm256_extracti128_si256(t2,1);
        __m128i t4 = _mm_add_epi32(_mm256_castsi256_si128(t2),t3);
        return static_cast<float>(_mm_cvtsi128_si32(t4));
    }
    float Avx2FloatHorizontalSum(__m256 a) {
        __m256 t1 = _mm256_hadd_ps(a,a);
        __m256 t2 = _mm256_hadd_ps(t1,t1);
        __m128 t3 = _mm256_extractf128_ps(t2,1);
        __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
        return _mm_cvtss_f32(t4);
    }
    float Avx2FloatHorizontalMax(__m256 x) {
        __m128 hi = _mm256_extractf128_ps(x, 1);
        __m128 lo = _mm256_extractf128_ps(x, 0);
        lo = _mm_max_ps(hi, lo);
        hi = _mm_movehl_ps(hi, lo);
        lo = _mm_max_ps(hi, lo);
        hi = _mm_shuffle_ps(lo, lo, 1);
        lo = _mm_max_ss(hi, lo);
        return _mm_cvtss_f32(lo);
    }
    float Avx2FloatHorizontalMin(__m256 x) {
        __m128 hi = _mm256_extractf128_ps(x, 1);
        __m128 lo = _mm256_extractf128_ps(x, 0);
        lo = _mm_min_ps(hi, lo);
        hi = _mm_movehl_ps(hi, lo);
        lo = _mm_min_ps(hi, lo);
        hi = _mm_shuffle_ps(lo, lo, 1);
        lo = _mm_min_ss(hi, lo);
        return _mm_cvtss_f32(lo);
    }
#endif
#ifdef USE_AVX512
    float Avx512IntHorizontalSum(__m512i a) {
        __m512i tmp = _mm512_add_epi32(a,_mm512_shuffle_i32x4(a,a,_MM_SHUFFLE(0,0,3,2)));
        __m128i r = _mm512_castsi512_si128(_mm512_add_epi32(tmp,_mm512_shuffle_i32x4(tmp,tmp,_MM_SHUFFLE(0,0,0,1))));
        r = _mm_hadd_epi32(r, r);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(r,r));
    }
    float Avx512FloatHorizontalSum(__m512 a) {
        __m512 tmp = _mm512_add_ps(a,_mm512_shuffle_f32x4(a,a,_MM_SHUFFLE(0,0,3,2)));
        __m128 r = _mm512_castps512_ps128(_mm512_add_ps(tmp,_mm512_shuffle_f32x4(tmp,tmp,_MM_SHUFFLE(0,0,0,1))));
        r = _mm_hadd_ps(r,r);
        return _mm_cvtss_f32(_mm_hadd_ps(r,r));
    }
#endif
    void vector_product(const float * start_a, const float * start_b, float & result,  int cnt){
#ifdef USE_SSE
        __m128 * as = (__m128*)(start_a);
        __m128 * bs = (__m128*)(start_b);
        int size = cnt * sizeof(float) / sizeof(__m128) / 2;
        __m128 res = _mm_setzero_ps();
        for (int i = 0; i < size; i++) {
            res = _mm_add_ps(res, _mm_add_ps(_mm_mul_ps(as[2*i], bs[2*i]), _mm_mul_ps( as[2*i+1], bs[2*i+1])));
        }
        res = _mm_hadd_ps(res, res);
        res = _mm_hadd_ps(res, res);
        _mm_store_ss(&result, res);
        for(int i = size*2*sizeof(__m128) / sizeof(float); i < cnt; i++){
            result += start_a[i] * start_b[i];
        }
#elif defined USE_AVX2
        __m256 res = _mm256_setzero_ps();
        for(int i = 0; i < cnt; i += sizeof(__m256)/sizeof(float)){
            auto a = _mm256_load_ps(&start_a[i]);
            auto b = _mm256_load_ps(&start_b[i]);
            res = _mm256_add_ps(res, _mm256_mul_ps(a, b));
        }
        result = Avx2FloatHorizontalSum(res);
#elif defined USE_AVX512
        __m512 res = _mm512_setzero_ps();
        for(int i = 0; i < cnt; i += sizeof(__m512)/sizeof(float)){
            auto a = _mm512_load_ps(&start_a[i]);
            auto b = _mm512_load_ps(&start_b[i]);
            res = _mm512_add_ps(res, _mm512_mul_ps(a, b));
        }
        result = _mm512_reduce_add_ps(res);
#else
        result = 0.0;
        for(int i = 0; i < cnt; ++i){
            result += start_a[i] * start_b[i];
        }
#endif
    }
    
    float quantized_vector_product(const size_t vectorSize,
                                   const unsigned char *quantizedInput,
                                   const char *weights) {
#ifdef USE_SSE
        // set sum to 0
        __m128i sum = _mm_setzero_si128();
        
        // loop for input_dimension times. (But with step size=16)
        // Because we quantized to 1 byte
        for (size_t j = 0; j < vectorSize; j += sizeof(__m128i)) {
            // load quantized unsigned char input values.
            const __m128i input128 = _mm_load_si128(reinterpret_cast<const __m128i *>(&quantizedInput[j]));
            const __m128i weight128 = _mm_load_si128(reinterpret_cast<const __m128i *>(&weights[j]));
            // c = saturate(i[0]*w[0]+i[1]*w[1]), saturate(i[2]*w[2]+i[3]*w[3]),...,
            // saturate(i[14]*w[14]+i[15]*w[15])
            // c contains eight 16 bit value.
            const __m128i c = _mm_maddubs_epi16(input128, weight128);
            // unpack 4 lowest 16 bit values to 32 bits.
            const __m128i lo = _mm_cvtepi16_epi32(c);
            // unpack 4 highest 16 bit values to 32 bits.
            const __m128i hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
            // add them to sum.
            sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
        }
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return static_cast<float>(_mm_extract_epi32(sum, 0));
#elif defined USE_AVX2
        __m256i sum = _mm256_setzero_si256();
        for (size_t j = 0; j < vectorSize; j += sizeof(__m256i)){
            const __m256i input = _mm256_load_si256(reinterpret_cast<const __m256i*>(&quantizedInput[j]));
            const __m256i weight = _mm256_load_si256(reinterpret_cast<const __m256i*>(&weights[j]));
            const __m256i c  = _mm256_maddubs_epi16(input, weight);
            const __m256i lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c, 0));
            const __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c, 1));
            sum = _mm256_add_epi32(sum, _mm256_add_epi32(lo, hi));
        }
        return Avx2IntHorizontalSum(sum);
#elif defined USE_AVX512
        __m512i sum = _mm512_setzero_si512();
        for (size_t j = 0; j < vectorSize; j += sizeof(__m512i)){
            const __m512i input = _mm512_load_si512(reinterpret_cast<const __m512i*>(&quantizedInput[j]));
            const __m512i weight = _mm512_load_si512(reinterpret_cast<const __m512i*>(&weights[j]));
            const __m512i c  = _mm512_maddubs_epi16(input, weight);
            const __m512i lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(c));
            const __m512i hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(c, 1));
            sum = _mm512_add_epi32(sum, _mm512_add_epi32(lo, hi));
        }
        return _mm512_reduce_add_epi32(sum);
#else
        float rst = 0.0;
        for(int j = 0; j < vectorSize; j++){
            rst += static_cast<unsigned int>(quantizedInput[j]) * static_cast<int>(weights[j]);
        }
        return rst;
#endif
    }
    float max_simd(const float *a, int n) {
#ifdef USE_SSE
        float res;
        
        __m128 *f4 = (__m128*) a;
        __m128 maxval = _mm_setzero_ps();
        
        for (int i = 0; i < n / 4; i++) {
            maxval = _mm_max_ps(maxval, f4[i]);
        }
        
        for (int i = 0; i < 3; i++) {
            maxval = _mm_max_ps(maxval, _mm_shuffle_ps(maxval, maxval, 0x93));
        }
        
        _mm_store_ss(&res, maxval);
        
        return res;
#elif defined USE_AVX2
        __m256 maxval = _mm256_setzero_ps();
        for(int i = 0; i < n; i += sizeof(__m256)/sizeof(float)){
            auto p = _mm256_load_ps(&a[i]);
            maxval = _mm256_max_ps(maxval, p);
        }
        return Avx2FloatHorizontalMax(maxval);
#elif defined USE_AVX512
        __m512 maxval = _mm512_setzero_ps();
        for(int i = 0; i < n; i += sizeof(__m512)/sizeof(float)){
            auto p = _mm512_load_ps(&a[i]);
            maxval = _mm512_max_ps(maxval, p);
        }
        return _mm512_reduce_max_ps(maxval);
#else
        float rst = std::numeric_limits<float>::min();
        for(int i = 0; i < n; ++i){
            rst = std::fmax(rst, a[i]);
        }
        return rst;
#endif
    }
    float min_simd(const float *a, int n) {
#ifdef USE_SSE
        float res;
        __m128 *f4 = (__m128*) a;
        __m128 minval = _mm_setzero_ps();
        for (int i = 0; i < n / 4; i++) {
            minval = _mm_min_ps(minval, f4[i]);
        }
        for (int i = 0; i < 3; i++) {
            minval = _mm_min_ps(minval, _mm_shuffle_ps(minval, minval, 0x93));
        }
        _mm_store_ss(&res, minval);
        return res;
#elif defined USE_AVX2
        __m256 minval = _mm256_setzero_ps();
        for(int i = 0; i < n; i += sizeof(__m256)/sizeof(float)){
            auto p = _mm256_load_ps(&a[i]);
            minval = _mm256_min_ps(minval, p);
        }
        return Avx2FloatHorizontalMin(minval);
#elif defined USE_AVX512
        __m512 minval = _mm512_setzero_ps();
        for(int i = 0; i < n; i += sizeof(__m512)/sizeof(float)){
            auto p = _mm512_load_ps(&a[i]);
            minval = _mm512_min_ps(minval, p);
        }
        return _mm512_reduce_min_ps(minval);
#else
        float rst = std::numeric_limits<float>::max();
        for(int i = 0; i < n; ++i){
            rst = std::fmin(rst, a[i]);
        }
        return rst;
#endif
    }
    template<typename T>
    void Convert(T* mO, unsigned out_stride, const float* mI, unsigned numRows, unsigned numCols, unsigned in_stride,  float scalingFctr, float biasFctr){
#ifdef USE_SSE
        unsigned ii, jj;
        __m128 a, b, c, d;
        __m128i ai, bi, ci, di, p0, p1;
        __m128 scalingFactor = _mm_set1_ps(scalingFctr);
        __m128 bias = _mm_set1_ps(biasFctr);
        const float *inptr;
        __m128i *outptr;
        for (ii = 0; ii < numRows; ii++) {
            inptr = &mI[ii * in_stride];
            outptr = (__m128i*)&mO[ii * out_stride];
            for (jj = 0; jj + SSE_STRIDE < numCols; jj += SSE_STRIDE) {
                a = _mm_load_ps(inptr);
                b = _mm_load_ps(inptr + 4);
                c = _mm_load_ps(inptr + 8);
                d = _mm_load_ps(inptr + 12);
                inptr += SSE_STRIDE;
                if(scalingFctr == 1.0f && biasFctr == 0.0f){
                    ai = _mm_cvtps_epi32(a);
                    bi = _mm_cvtps_epi32(b);
                    ci = _mm_cvtps_epi32(c);
                    di = _mm_cvtps_epi32(d);
                }else if(biasFctr == 0.0f){
                    ai = _mm_cvtps_epi32(_mm_mul_ps(a, scalingFactor));
                    bi = _mm_cvtps_epi32(_mm_mul_ps(b, scalingFactor));
                    ci = _mm_cvtps_epi32(_mm_mul_ps(c, scalingFactor));
                    di = _mm_cvtps_epi32(_mm_mul_ps(d, scalingFactor));
                }else if(scalingFctr == 1.0f){
                    ai = _mm_cvtps_epi32(_mm_add_ps(a, bias));
                    bi = _mm_cvtps_epi32(_mm_add_ps(b, bias));
                    ci = _mm_cvtps_epi32(_mm_add_ps(c, bias));
                    di = _mm_cvtps_epi32(_mm_add_ps(d, bias));
                }else{
                    ai = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(a, scalingFactor), bias));
                    bi = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(b, scalingFactor), bias));
                    ci = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(c, scalingFactor), bias));
                    di = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(d, scalingFactor), bias));
                }
                p0 = _mm_packs_epi32(ai, bi);
                p1 = _mm_packs_epi32(ci, di);
                if(std::is_same<T, char>::value){
                    p0 = _mm_packs_epi16(p0, p1);
                }else if(std::is_same<T, unsigned char>::value){
                    p0 = _mm_packus_epi16(p0, p1);
                }
                _mm_store_si128(outptr++, p0);
            }
            for(;jj < numCols; ++jj){
                mO[ii * out_stride + jj] = SaturateInt<T>(round(mI[ii * in_stride + jj] * scalingFctr + biasFctr));
            }
        }
#elif defined USE_AVX2
        unsigned ii, jj;
        __m256 a, b, c, d;
        __m256i ai, bi, ci, di, p0, p1;
        __m256 scalingFactor = _mm256_set1_ps(scalingFctr);
        __m256 bias = _mm256_set1_ps(biasFctr);
        const float *inptr;
        __m256i *outptr;
        for (ii = 0; ii < numRows; ii++) {
            inptr = &mI[ii * in_stride];
            outptr = (__m256i*)&mO[ii * out_stride];
            for (jj = 0; jj + AVX2_STRIDE <= numCols; jj += AVX2_STRIDE) {
                a = _mm256_load_ps(inptr);
                b = _mm256_load_ps(inptr + 8);
                c = _mm256_load_ps(inptr + 16);
                d = _mm256_load_ps(inptr + 24);
                inptr += AVX2_STRIDE;
                
                if(scalingFctr == 1.0f && biasFctr == 0.0f){
                    ai = _mm256_cvtps_epi32(a);
                    bi = _mm256_cvtps_epi32(b);
                    ci = _mm256_cvtps_epi32(c);
                    di = _mm256_cvtps_epi32(d);
                }else if(biasFctr == 0.0f){
                    ai = _mm256_cvtps_epi32(_mm256_mul_ps(a, scalingFactor));
                    bi = _mm256_cvtps_epi32(_mm256_mul_ps(b, scalingFactor));
                    ci = _mm256_cvtps_epi32(_mm256_mul_ps(c, scalingFactor));
                    di = _mm256_cvtps_epi32(_mm256_mul_ps(d, scalingFactor));
                }else if(scalingFctr == 1.0f){
                    ai = _mm256_cvtps_epi32(_mm256_add_ps(a, bias));
                    bi = _mm256_cvtps_epi32(_mm256_add_ps(b, bias));
                    ci = _mm256_cvtps_epi32(_mm256_add_ps(c, bias));
                    di = _mm256_cvtps_epi32(_mm256_add_ps(d, bias));
                }else{
                    ai = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(a, scalingFactor), bias));
                    bi = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(b, scalingFactor), bias));
                    ci = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(c, scalingFactor), bias));
                    di = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(d, scalingFactor), bias));
                }
                
                p0 = _mm256_packs_epi32(ai, bi);
                p0 = _mm256_permute4x64_epi64(p0, 0xd8);
                p1 = _mm256_packs_epi32(ci, di);
                p1 = _mm256_permute4x64_epi64(p1, 0xd8);
                if(std::is_same<T, char>::value){
                    p0 = _mm256_packs_epi16(p0, p1);
                }else if(std::is_same<T, unsigned char>::value){
                    p0 = _mm256_packus_epi16(p0, p1);
                }
                p0 = _mm256_permute4x64_epi64(p0, 0xd8);
                _mm256_store_si256(outptr++, p0);
            }
            for(;jj < numCols; ++jj){
                mO[ii * out_stride + jj] = SaturateInt<T>(round(mI[ii * in_stride + jj] * scalingFctr + biasFctr));
            }
        }
#elif defined USE_AVX512
        unsigned ii, jj;
        __m512 a, b, c, d;
        __m512i ai, bi, ci, di, p0, p1;
        __m512 scalingFactor = _mm512_set1_ps(scalingFctr);
        __m512 bias = _mm512_set1_ps(biasFctr);
        const float *inptr;
        __m512i *outptr;
        for (ii = 0; ii < numRows; ii++) {
            inptr = &mI[ii * in_stride];
            outptr = (__m512i*)&mO[ii * out_stride];
            for (jj = 0; jj + AVX512_STRIDE < numCols; jj += AVX512_STRIDE) {
                a = _mm512_load_ps(inptr);
                b = _mm512_load_ps(inptr + 16);
                c = _mm512_load_ps(inptr + 32);
                d = _mm512_load_ps(inptr + 48);
                inptr += AVX512_STRIDE;
                ai = _mm512_cvtps_epi32(_mm512_mul_ps(a, scalingFactor));
                bi = _mm512_cvtps_epi32(_mm512_mul_ps(b, scalingFactor));
                ci = _mm512_cvtps_epi32(_mm512_mul_ps(c, scalingFactor));
                di = _mm512_cvtps_epi32(_mm512_mul_ps(d, scalingFactor));
                
                if(scalingFctr == 1.0f && biasFctr == 0.0f){
                    ai = _mm512_cvtps_epi32(a);
                    bi = _mm512_cvtps_epi32(b);
                    ci = _mm512_cvtps_epi32(c);
                    di = _mm512_cvtps_epi32(d);
                }else if(biasFctr == 0.0f){
                    ai = _mm512_cvtps_epi32(_mm512_mul_ps(a, scalingFactor));
                    bi = _mm512_cvtps_epi32(_mm512_mul_ps(b, scalingFactor));
                    ci = _mm512_cvtps_epi32(_mm512_mul_ps(c, scalingFactor));
                    di = _mm512_cvtps_epi32(_mm512_mul_ps(d, scalingFactor));
                }else if(scalingFctr == 1.0f){
                    ai = _mm512_cvtps_epi32(_mm512_add_ps(a, bias));
                    bi = _mm512_cvtps_epi32(_mm512_add_ps(b, bias));
                    ci = _mm512_cvtps_epi32(_mm512_add_ps(c, bias));
                    di = _mm512_cvtps_epi32(_mm512_add_ps(d, bias));
                }else{
                    ai = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(a, scalingFactor), bias));
                    bi = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(b, scalingFactor), bias));
                    ci = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(c, scalingFactor), bias));
                    di = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(d, scalingFactor), bias));
                }
                p0 = _mm512_packs_epi32(ai, bi);
                p0 = _mm512_permutex_epi64(p0, 0xd8);
                p0 = _mm512_shuffle_i32x4(p0, p0, 0xd8);
                
                p1 = _mm512_packs_epi32(ci, di);
                p1 = _mm512_permutex_epi64(p1, 0xd8);
                p1 = _mm512_shuffle_i32x4(p1, p1, 0xd8);
                
                
                if(std::is_same<T, char>::value){
                    p0 = _mm512_packs_epi16(p0, p1);
                }else if(std::is_same<T, unsigned char>::value){
                    p0 = _mm512_packus_epi16(p0, p1);
                }
                p0 = _mm512_permutex_epi64(p0, 0xd8);
                p0 = _mm512_shuffle_i32x4(p0, p0, 0xd8);
                _mm512_store_si512(outptr++, p0);
            }
            for(;jj < numCols; ++jj){
                mO[ii * out_stride + jj] = SaturateInt<T>(round(mI[ii * in_stride + jj] * scalingFctr + biasFctr));
            }
        }
#else
        unsigned ii, jj;
        for (ii = 0; ii < numRows; ii++) {
            const float* inptr = &mI[ii * in_stride];
            T* outptr = &mO[ii * out_stride];
            for (jj = 0; jj < numCols; jj ++) {
                outptr[jj] = SaturateInt<T>(round(inptr[jj] * scalingFctr + biasFctr));
            }
        }
#endif
    }
    
    void multiplyMat(float* x, const float* t, int size){
#ifdef USE_SSE
        for(int i = 0; i < size; i += sizeof(__m128)/sizeof(float)){
            _mm_store_ps(&x[i], _mm_mul_ps(_mm_load_ps(&x[i]), _mm_load_ps(&t[i])));
        }
#elif defined USE_AVX2
        for(int i = 0; i < size; i += sizeof(__m256)/sizeof(float)){
            _mm256_store_ps(&x[i], _mm256_mul_ps(_mm256_load_ps(&x[i]), _mm256_load_ps(&t[i])));
        }
#elif defined USE_AVX512
        for(int i = 0; i < size; i += sizeof(__m512)/sizeof(float)){
            _mm512_store_ps(&x[i], _mm512_mul_ps(_mm512_load_ps(&x[i]), _mm512_load_ps(&t[i])));
        }
#else
        for(int i = 0; i < size; i++){
            x[i] *= t[i];
        }
#endif
    }
    
    void multiplyMat(float* x, const float* t1, const float* t2, int size){
#ifdef USE_SSE
        for(int i = 0; i < size; i += sizeof(__m128)/sizeof(float)){
            _mm_store_ps(&x[i], _mm_mul_ps(_mm_load_ps(&t1[i]), _mm_load_ps(&t2[i])));
        }
#elif defined USE_AVX2
        for(int i = 0; i < size; i += sizeof(__m256)/sizeof(float)){
            _mm256_store_ps(&x[i], _mm256_mul_ps(_mm256_load_ps(&t1[i]), _mm256_load_ps(&t2[i])));
        }
#elif defined USE_AVX512
        for(int i = 0; i < size; i += sizeof(__m512)/sizeof(float)){
            _mm512_store_ps(&x[i], _mm512_mul_ps(_mm512_load_ps(&t1[i]), _mm512_load_ps(&t2[i])));
        }
#else
        for(int i = 0; i < size; i++){
            x[i] = t1[i]*t2[i];
        }
#endif
    }
    void addMat(float* x, const float* t, int size){
#ifdef USE_SSE
        for(int i = 0; i < size; i += sizeof(__m128)/sizeof(float)){
            _mm_store_ps(&x[i], _mm_add_ps(_mm_load_ps(&x[i]), _mm_load_ps(&t[i])));
        }
#elif defined USE_AVX2
        for(int i = 0; i < size; i += sizeof(__m256)/sizeof(float)){
            _mm256_store_ps(&x[i], _mm256_add_ps(_mm256_load_ps(&x[i]), _mm256_load_ps(&t[i])));
        }
#elif defined USE_AVX512
        for(int i = 0; i < size; i += sizeof(__m512)/sizeof(float)){
            _mm512_store_ps(&x[i], _mm512_add_ps(_mm512_load_ps(&x[i]), _mm512_load_ps(&t[i])));
        }
#else
        for(int i = 0; i < size; i++){
            x[i] += t[i];
        }
#endif
    }
    
    void addMat(float* x, const float* t1, const float* t2, int size){
#ifdef USE_SSE
        for(int i = 0; i < size; i += sizeof(__m128)/sizeof(float)){
            _mm_store_ps(&x[i], _mm_add_ps(_mm_load_ps(&t1[i]), _mm_load_ps(&t2[i])));
        }
#elif defined USE_AVX2
        for(int i = 0; i < size; i += sizeof(__m256)/sizeof(float)){
            _mm256_store_ps(&x[i], _mm256_add_ps(_mm256_load_ps(&t1[i]), _mm256_load_ps(&t2[i])));
        }
#elif defined USE_AVX512
        for(int i = 0; i < size; i += sizeof(__m512)/sizeof(float)){
            _mm512_store_ps(&x[i], _mm512_add_ps(_mm512_load_ps(&t1[i]), _mm512_load_ps(&t2[i])));
        }
#else
        for(int i = 0; i < size; i++){
            x[i] = t1[i] + t2[i];
        }
#endif
    }
    void addMatBias(float* x, const float* t, const float* bias, int size){
#ifdef USE_SSE
        for(int i = 0; i < size; i += sizeof(__m128)/sizeof(float)){
            _mm_store_ps(&x[i], _mm_add_ps(_mm_load_ps(&x[i]) ,_mm_add_ps(_mm_load_ps(&t[i]), _mm_load_ps(&bias[i]))));
        }
#elif defined USE_AVX2
        for(int i = 0; i < size; i += sizeof(__m256)/sizeof(float)){
            _mm256_store_ps(&x[i], _mm256_add_ps(_mm256_load_ps(&x[i]), _mm256_add_ps(_mm256_load_ps(&t[i]), _mm256_load_ps(&bias[i]))));
        }
#elif defined USE_AVX512
        for(int i = 0; i < size; i += sizeof(__m512)/sizeof(float)){
            _mm512_store_ps(&x[i], _mm512_add_ps(_mm512_load_ps(&x[i]), _mm512_add_ps(_mm512_load_ps(&t[i]), _mm512_load_ps(&bias[i]))));
        }
#else
        for(int i = 0; i < size; i++){
            x[i] += t[i] + bias[i];
        }
#endif
    }
    void relu(const float* from, float* to, int size){
#ifdef USE_SSE
        for(int k = 0; k < size; k += sizeof(__m128)/sizeof(float)){
            _mm_store_ps(&to[k], _mm_max_ps(_mm_load_ps(&from[k]), _mm_set1_ps(0.0f)));
        }
#elif defined USE_AVX2
        for(int k = 0; k < size; k += sizeof(__m256)/sizeof(float)){
            _mm256_store_ps(&to[k], _mm256_max_ps(_mm256_load_ps(&from[k]), _mm256_set1_ps(0.0f)));
        }
#elif defined USE_AVX512
        for(int k = 0; k < size; k += sizeof(__m512)/sizeof(float)){
            _mm512_store_ps(&to[k], _mm512_max_ps(_mm512_load_ps(&from[k]), _mm512_set1_ps(0.0f)));
        }
#else
        for(int j = 0 ;j < size; ++j){
            to[j] = fmax(from[j], 0);
        }
#endif
    }
    void vector_scale(const float* from, float* to, float scale, int size){
#ifdef USE_SSE
        for(int k = 0; k < size; k += sizeof(__m128)/sizeof(float)){
            _mm_store_ps(&to[k], _mm_mul_ps(_mm_load_ps(&from[k]), _mm_set1_ps(scale)));
        }
#elif defined USE_AVX2
        for(int k = 0; k < size; k += sizeof(__m256)/sizeof(float)){
            _mm256_store_ps(&to[k], _mm256_mul_ps(_mm256_load_ps(&from[k]), _mm256_set1_ps(scale)));
        }
#elif defined USE_AVX512
        for(int k = 0; k < size; k += sizeof(__m512)/sizeof(float)){
            _mm512_store_ps(&to[k], _mm512_mul_ps(_mm512_load_ps(&from[k]), _mm512_set1_ps(scale)));
        }
#else
        for(int j = 0 ;j < size; ++j){
            to[j] = from[j] * scale;
        }
#endif
    }
    void logMat(float* x, int row, int col, int stride){
/*
#ifdef USE_SSE
        for(int i = 0; i < row; ++i){
            for(int k = 0; k < stride; k += sizeof(__m128)/sizeof(float)){
                _mm_store_ps(&x[stride*i + k], log_ps(_mm_add_ps(_mm_load_ps(&x[stride*i + k]), _mm_set1_ps(1e-20))));
            }
        }
#elif defined(USE_AVX2) || defined(USE_AVX512)
        for(int i = 0; i < row; ++i){
            for(int k = 0; k < stride; k += sizeof(__m256)/sizeof(float)){
                _mm256_store_ps(&x[stride*i + k], log256_ps(_mm256_add_ps(_mm256_load_ps(&x[stride*i + k]), _mm256_set1_ps(1e-20))));
            }
        }
*/
#if defined USE_CUDA
        ApplyLog(x, row, col, stride);
#else
        for(int i = 0; i < row; ++i){
            for(int k = 0; k < col; ++k){
                x[stride*i + k] = log(x[stride*i + k] + 1e-20);
            }
        }
#endif
    }
     // below now are deperated...
#if defined USE_SSE || defined USE_AVX2 || defined USE_AVX512
     inline void _mm_madd_epi8_SSE(const __m128i & a, const __m128i & b,
         __m128i& madd_epi32_l, __m128i& madd_epi32_h){
         // a = 0x00 0x01 0xFE 0x04 ...
         // b = 0x00 0x02 0x80 0x84 ...
         // To extend signed 8-bit value, MSB has to be set to 0xFF
         __m128i sign_mask_a  = _mm_cmplt_epi8(a, _mm_setzero_si128());
         __m128i sign_mask_b  = _mm_cmplt_epi8(b, _mm_setzero_si128());
         
         // sign_mask_a = 0x00 0x00 0xFF 0x00 ...
         // sign_mask_b = 0x00 0x00 0xFF 0xFF ...
         
         // Unpack positives with 0x00, negatives with 0xFF
         __m128i a_epi16_l    = _mm_unpacklo_epi8(a, sign_mask_a);
         __m128i a_epi16_h    = _mm_unpackhi_epi8(a, sign_mask_a);
         __m128i b_epi16_l    = _mm_unpacklo_epi8(b, sign_mask_b);
         __m128i b_epi16_h    = _mm_unpackhi_epi8(b, sign_mask_b);
         
         // Here - valid 16-bit signed integers corresponding to the 8-bit input
         // a_epi16_l = 0x00 0x00 0x01 0x00 0xFE 0xFF 0x04 0x00 ...
         
         // Get the a[i] * b[i] + a[i+1] * b[i+1] for both low and high parts
         madd_epi32_l = _mm_madd_epi16(a_epi16_l, b_epi16_l);
         madd_epi32_h = _mm_madd_epi16(a_epi16_h, b_epi16_h);
     }
     
     inline float quantized_vector_product(const size_t vectorSize,
                                           const char *quantizedInput,
                                           const char *weights) {
         // set sum to 0
         __m128i sum = _mm_setzero_si128();
         for (size_t j = 0; j < vectorSize; j += 16) {
             // load quantized unsigned char input values.
             const __m128i input128 = _mm_load_si128(reinterpret_cast<const __m128i *>(&quantizedInput[j]));
             const __m128i weight128 = _mm_load_si128(reinterpret_cast<const __m128i *>(&weights[j]));
             __m128i madd_epi32_l, madd_epi32_h;
             _mm_madd_epi8_SSE(input128, weight128, madd_epi32_l, madd_epi32_h);
             sum = _mm_add_epi32(_mm_add_epi32(madd_epi32_l, madd_epi32_h), sum);
         }
         sum = _mm_hadd_epi32(sum, sum);
         sum = _mm_hadd_epi32(sum, sum);
         return static_cast<float>(_mm_extract_epi32(sum, 0));
     }
#endif
    template
    void Convert<char>(char* mO, unsigned out_stride, const float* mI, unsigned numRows, unsigned numCols, unsigned in_stride,  float scalingFctr, float biasFctr);
    template
    void Convert<unsigned char>(unsigned char* mO, unsigned out_stride, const float* mI, unsigned numRows, unsigned numCols, unsigned in_stride,  float scalingFctr, float biasFctr);
    template
    void Convert<float>(float* mO, unsigned out_stride, const float* mI, unsigned numRows, unsigned numCols, unsigned in_stride,  float scalingFctr, float biasFctr);
}
