

#ifndef nnet_matrix_h
#define nnet_matrix_h
#include "nnet-comm.h"
#include "nnet-io.h"
#include <limits>
#include <atomic>
#include <cassert>
namespace DNN{
    class MatrixBaseBase{
    public:
        static int Skip();
        static void SetSkip(int i);
    protected:
        // CAUTION : skip is not guaranteed to be thread safe
        // we make skip as static because we consider it as a global varible.
        // we do not support doing frame skipping with different skip values at the same time for different threads.
        static std::atomic<int> skip;
    }; 
    template<typename T> class SubMatrix;
    template<typename T>
    class MatrixBase : public MatrixBaseBase{
    public:
        friend class SubMatrix<T>;
        MatrixBase(const MatrixBase&) = delete; 
        void operator=(const MatrixBase&) = delete;
#ifndef USE_CUDA
        T& operator()(int r,int c){
            assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
            return *(data_ + r * stride_ + c );
        }
        T operator()(int r,int c) const {
            LOG_ASSERT(r >= 0 && r < rows_ && c >= 0 && c < cols_);
            return *(data_ + r * stride_ + c );
        }
#endif
        T* Data();
        const T* Data() const;
        const int NumRows() const { return rows_; }
        const int NumCols() const { return cols_; }
        const int Stride() const {return stride_;}
        const Quantizer<T>& GetQuantizer() const {return quantizer_;}
        void SetQuantizer(const Quantizer<T>& multi){ quantizer_ = multi;}
        T* RowData(const int idx) ;
        const T* RowData(const int idx) const;
        void SetZero();
        T MaxAbs(T t1, T t2) const;
        void Swap(MatrixBase<T> *other);
        void Write(std::ostream & os, bool binary) const;
    protected:
        MatrixBase(): rows_(0), cols_(0), stride_(0), malloc_rows_(0), data_(nullptr){ }
        MatrixBase(T *data,
                    int num_rows,
                    int num_cols,
                    int stride):rows_(num_rows), cols_(num_cols), stride_(stride), malloc_rows_(num_rows), data_(data) { }
        int rows_ = 0;
        int cols_ = 0;
        int stride_ = 0;
        int malloc_rows_ = 0;
        T* data_ = nullptr;
        Quantizer<T> quantizer_;
    };
    template<typename T>
    class Matrix : public MatrixBase<T>{
    public:
        friend class SubMatrix<T>;
        Matrix(){}
        Matrix(const std::vector<std::vector<T>>& V, bool transpose = false);
        Matrix(const std::vector<T>& V, int row, int col);
        Matrix(const Matrix<T>& other, int row);
        Matrix(int rows, int cols);
        ~Matrix();
        Matrix (const Matrix<T> & M);
        Matrix (const MatrixBase<T> & M);
        Matrix<T> & operator = (const Matrix<T> & M);
        void Resize(int row, int col);
        void ReadFromVec(const std::vector<std::vector<T>>& V, bool transpose = false);
        void ReshapeFromVec(const std::vector<T>& V, int row, int col);
        void CopyFromMat(const Matrix<T> &other);
        void Read(std::istream & is, bool binary);
    private:
        void Release();
    };
    template<typename T>
    class SubMatrix : public MatrixBase<T>{
    public:
        inline SubMatrix(const MatrixBase<T> &mat,
                         const unsigned row_offset,
                         const unsigned num_rows,
                         const unsigned col_offset,
                         const unsigned num_cols){
            if (num_rows == 0 || num_cols == 0) {
                LOG_ASSERT(num_rows == 0 && num_cols == 0);

            } else {
                LOG_ASSERT(row_offset >= 0 && col_offset >= 0 &&
                             row_offset + num_rows <= mat.rows_ &&
                             col_offset + num_cols <= mat.cols_);
                this->data_ = mat.data_ + static_cast<size_t>(col_offset) +
                    static_cast<size_t>(row_offset) * static_cast<size_t>(mat.stride_);
                this->cols_ = num_cols;
                this->rows_ = num_rows;
                this->stride_ = mat.stride_;
                this->quantizer_ = mat.quantizer_;
            }
        }
        inline SubMatrix<T> (const SubMatrix &other):
        MatrixBase<T> (other.data_, other.num_rows_, other.num_cols_,
                      other.stride_) {
            this->quantizer_ = other.quantizer_;
        }
    private:
        SubMatrix<T> &operator = (const SubMatrix<T> &other);
    };
    
    
}// end of namespace DNN

#include "nnet-matrix-imp.h"
#endif /* nnet_matrix_h */
