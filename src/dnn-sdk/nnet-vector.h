

#ifndef nnet_vector_h
#define nnet_vector_h
#include "nnet-comm.h"
#include "nnet-io.h"
#include "nnet-matrix.h"
#include <limits>
namespace DNN{
    template<typename T>
    class Vector{
    public:
        Vector();
        Vector(const std::vector<T>& V);
        Vector (const Vector<T> & V);
        Vector (const MatrixBase<T>& Mat, int row);
        Vector<T> & operator = (const Vector<T> & V);
        ~Vector();
        void Write(std::ostream & os, bool binary) const;
        void Read(std::istream & is, bool binary);
        void Resize(int dim);
        void SetZero();
        void Release();
        T* Data() {return data_;}
        const T* Data() const {return data_;}
        
        const int Dim() const {return dim_;}
        const int Stride() const {return stride_;}
#ifndef USE_CUDA 
        T& operator()(int r){
            LOG_ASSERT(r >= 0 && r < dim_);
            return *(data_ + r);
        }
        T operator()(int r) const{
            LOG_ASSERT(r >= 0 && r < dim_);
            return *(data_ + r);
        }
#endif
        void CopyFromVec(const Vector<T> &other);
        void ReadFromVec(const std::vector<T>& V);
    protected:
        int dim_ = 0;
        T* data_ = nullptr;
        int stride_ = 0;
    };
} // end of namespace DNN
#include "nnet-vector-imp.h"

#endif /* nnet_vector_h */
