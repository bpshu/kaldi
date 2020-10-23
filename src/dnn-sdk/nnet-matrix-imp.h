

#ifndef nnet_matrix_imp_h
#define nnet_matrix_imp_h

namespace DNN{
    
    template<typename T>
    T* MatrixBase<T>::Data(){
        return data_;
    }
    template<typename T>
    const T* MatrixBase<T>::Data() const{
        return data_;
    }
    template<typename T>
    T* MatrixBase<T>::RowData(const int idx) {
        LOG_ASSERT(idx < rows_ && idx >= 0);
        return data_ + stride_ * idx;
    }
    template<typename T>
    const T* MatrixBase<T>::RowData(const int idx) const{
        assert(idx < rows_ && idx >= 0);
        return data_ + stride_ * idx;
    }
    template<typename T>
    void MatrixBase<T>::SetZero(){
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMemset2D(data_, stride_ * sizeof(T), 0,
                                  cols_ * sizeof(T), rows_ ));
#else
        if(data_) memset(data_, 0, sizeof(T)*rows_*stride_);
#endif
    }
    template<typename T>
    T MatrixBase<T>::MaxAbs(T t1, T t2) const{
#ifdef USE_CUDA
        LOG_ASSERT(0 && "MaxAbs not supported in GPU");
#endif
        T max_v = 0;
        T tmp = 0;
        for(int i = 0 ; i < rows_; i ++ ){
            for(int j = 0 ; j < cols_ ; j++){
                tmp = *(data_ + i * stride_ + j );
                tmp = tmp > 0 ? tmp : 0 - tmp;
                if(tmp > t2) tmp = t2;
                if(tmp < t1) tmp = t1;
                if(tmp > max_v) max_v = tmp;
            }
        }
        return max_v;
    }
    template<typename T>
    void MatrixBase<T>::Swap(MatrixBase<T> *other){
        std::swap(this->data_, other->data_);
        std::swap(this->cols_, other->cols_);
        std::swap(this->rows_, other->rows_);
        std::swap(this->stride_, other->stride_);
        std::swap(this->malloc_rows_, other->malloc_rows_);
        std::swap(this->quantizer_, other->quantizer_);
    }

    template<typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T>>& V, bool transpose){
        this->ReadFromVec(V, transpose);
    }
    template<typename T>
    Matrix<T>::Matrix(const std::vector<T>& V, int row, int col){
        this->ReshapeFromVec(V, row, col);
    }
    template<typename T>
    Matrix<T>::Matrix(const Matrix<T> & M){
        this->CopyFromMat(M);
    }
    template<typename T>
    Matrix<T>::Matrix(const MatrixBase<T> & M){
        this->CopyFromMat(M);
    }
    template<typename T>
    Matrix<T>::Matrix(const Matrix<T>& other, int row){
        this->Release();
        this->Resize(1, other.NumCols());
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMemcpy(this->RowData(0), other.RowData(row), sizeof(T)*other.Stride(), cudaMemcpyDeviceToDevice));
#else
        memcpy(this->RowData(0), other.RowData(row), sizeof(T)*other.Stride());
#endif
    }
    template<typename T>
    Matrix<T>& Matrix<T>::operator =(const Matrix<T> & M){
        this->CopyFromMat(M);
        return *this;
    }

    template<typename T>
    Matrix<T>::~Matrix(){
        this->Release();
    }
    template<typename T>
    Matrix<T>::Matrix(int rows, int cols){
        this->Resize(rows, cols);
    }
    template<typename T>
    void Matrix<T>::Release(){
        if(this->data_)  {
            aligned_free(this->data_);
            this->data_ = nullptr;
        }
        this->rows_ = this->cols_ = this->stride_ = this->malloc_rows_ = 0;
    }
    template<typename T>
    void Matrix<T>::Resize(int row, int col){
        if(row == this->rows_ && col == this->cols_) {
            this->SetZero();
            return;
        }
        if(row < 0 || col < 0){ LOG_ERROR("error row or col"); }
        if(row == 0 || col == 0){
            this->Release();
            this->SetZero();
            return;
        }
        if(this->malloc_rows_ >= row && this->stride_  >= col){
            this->rows_ = row;
            this->cols_ = col;
            this->SetZero();
            return;
        }
        //else free first and malloc again
        this->Release();
#ifdef USE_CUDA
        size_t pitch;
        CU_SAFE_CALL(cudaMallocPitch((void**)&(this->data_), &pitch, col * sizeof(T), row));
        this->rows_ = row;
        this->cols_ = col;
        this->stride_ = pitch / sizeof(T);
#else
        int skip = ((ALIGN / sizeof(T)) - col % (ALIGN / sizeof(T))) % (ALIGN / sizeof(T));
        this->stride_ = col + skip;
        this->cols_ = col;
        this->malloc_rows_ = this->rows_ = row;
        this->data_ = AlignedAlloc<T>(this->stride_ * this->malloc_rows_);
        if (!this->data_){
            throw std::bad_alloc();
        }
#endif
        this->SetZero();
    }

    template<typename T>
    void Matrix<T>::CopyFromMat(const Matrix<T>& b){
        if(static_cast<const void*>(this->Data()) ==
           static_cast<const void*>(b.Data())){
            if (b.Data() == nullptr)
                return;
            LOG_ASSERT(b.NumRows() == this->NumRows() && b.NumCols() == this->NumCols() && b.Stride() == this->Stride());
            return;
        }
        this->Resize(b.NumRows(),b.NumCols());
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMemcpy2D(this->Data(), this->Stride()*sizeof(T), b.Data(), b.Stride()*sizeof(T),
                                  b.NumCols() * sizeof(T), b.NumRows(), cudaMemcpyDeviceToDevice));
#else
        for(int i = 0; i < this->rows_; i++){
            memcpy(this->RowData(i), b.RowData(i), sizeof(T) * this->stride_);
        }
#endif
        this->quantizer_ = b.quantizer_;
    }

    template<typename T>
    void Matrix<T>::ReadFromVec(const std::vector<std::vector<T>>& V, bool transpose){
        LOG_ASSERT(!V.empty());
        LOG_ASSERT(!V[0].empty());
        if(!transpose){
            this->Resize(V.size(), V[0].size());
#ifdef USE_CUDA
            for(int i = 0; i < this->NumRows(); ++i){
                CU_SAFE_CALL(cudaMemcpy(this->RowData(i), V[i].data(), V[0].size()*sizeof(T), cudaMemcpyHostToDevice));
            }
#else
            for(int i = 0; i < this->NumRows(); ++i){
                for(int j = 0; j < this->NumCols(); ++j){
                    (*this)(i, j) = V[i][j];
                }
            }
#endif
        }else{
#ifdef USE_CUDA
            LOG_ASSERT(0 && "transpose not supported in GPU");
#else
            this->Resize(V[0].size(), V.size());
            for(int i = 0; i < this->NumRows(); ++i){
                for(int j = 0; j < this->NumCols(); ++j){
                    (*this)(i, j) = V[j][i];
                }
            }
#endif
        }
    }
    template<typename T>
    void Matrix<T>::ReshapeFromVec(const std::vector<T>& V, int row, int col){
        LOG_ASSERT(V.size() == row * col);
        this->Resize(row, col);
        for(int i = 0; i < this->NumRows(); ++i){
#ifdef USE_CUDA
            CU_SAFE_CALL(cudaMemcpy(this->RowData(i), V.data() + i*col, col*sizeof(T), cudaMemcpyHostToDevice));
#else
            for(int j = 0; j < this->NumCols(); ++j){
                (*this)(i,j) = V[i * col + j];
            }
#endif
        }
    }

    template<typename T>
    void Matrix<T>::Read(std::istream & is, bool binary){
        std::ostringstream specific_error;
        using MatrixIndexT = int32_t;
        if(binary){
            uint32_t rows = ReadUint32(is);
            uint32_t cols = ReadUint32(is);
            this->Resize(rows, cols);
#ifdef USE_CUDA
            T* tmp = new T[cols];
            for (MatrixIndexT i = 0; i < (MatrixIndexT)rows; i++) {
                is.read(reinterpret_cast<char*>(tmp), sizeof(T)*cols);
                CU_SAFE_CALL(cudaMemcpy(this->RowData(i), tmp, cols*sizeof(T), cudaMemcpyHostToDevice));
                if (is.fail()) goto bad;
            }
            delete []tmp;
#else
            for (MatrixIndexT i = 0; i < (MatrixIndexT)rows; i++) {
                is.read(reinterpret_cast<char*>(this->RowData(i)), sizeof(T)*cols);
                if (is.fail()) goto bad;
            }
#endif
            if (is.eof()) return;
            if (is.fail()) goto bad;
            return;
        }else{
#ifdef USE_CUDA
            LOG_ASSERT(0 && "Matrix Read() in non-binary format is not supported in GPU");
#else
            std::string str;
            is >> str;
            if (is.fail()) {
                LOG_ERROR("Failed to read matrix from stream");
            }
            if (str == "[]") {
                Resize(0, 0);
                return;
            }else if (str != "[") {
                LOG_ERROR("Expected \"[\", got \""+ str + "\"");
            }
            // At this point, we have read "[".
            std::vector<std::vector<T>* > data;
            std::vector<T> *cur_row = new std::vector<T>;
            while (1) {
                int i = is.peek();
                if (i == -1) {
                    LOG_WARN("Got EOF while reading matrix data");
                    goto cleanup;
                }
                else if (static_cast<char>(i) == ']') {  // Finished reading matrix.
                    is.get();  // eat the "]".
                    i = is.peek();
                    if (static_cast<char>(i) == '\r') {
                        is.get();
                        is.get();  // get \r\n (must eat what we wrote)
                    } else if (static_cast<char>(i) == '\n') {
                        is.get();
                        // get \n (must eat what we wrote)
                    }
                    if (is.fail()) {
                        LOG_ERROR("After end of matrix data, read error.");
                        // we got the data we needed, so just warn for this error.
                    }
                    // Now process the data.
                    if (!cur_row->empty()) data.push_back(cur_row);
                    else delete(cur_row);
                    cur_row = NULL;
                    if (data.empty()) { this->Resize(0, 0); return; }
                    else {
                        int32_t num_rows = data.size(), num_cols = data[0]->size();
                        this->Resize(num_rows, num_cols);
                        for (int32_t i = 0; i < num_rows; i++) {
                            if (static_cast<int32_t>(data[i]->size()) != num_cols) {
                                LOG_WARN ("Matrix has inconsistent #cols: " << num_cols
                                << " vs." << data[i]->size() << " (processing row"
                                << i << ")");
                                goto cleanup;
                            }
                            for (int32_t j = 0; j < num_cols; j++)
                                (*this)(i, j) = (*(data[i]))[j];
                            delete data[i];
                            data[i] = NULL;
                        }
                    }
                    return;
                } else if (static_cast<char>(i) == '\n' || static_cast<char>(i) == ';') {
                    // End of matrix row.
                    is.get();
                    if (cur_row->size() != 0) {
                        data.push_back(cur_row);
                        cur_row = new std::vector<T>;
                        cur_row->reserve(data.back()->size());
                    }
                } else if ( (i >= '0' && i <= '9') || i == '-' ) {  // A number...
                    T r;
                    is >> r;
                    if (is.fail()) {
                        LOG_WARN("Stream failure/EOF while reading matrix data.");
                        goto cleanup;
                    }
                    cur_row->push_back(r);
                } else if (isspace(i)) {
                    is.get();  // eat the space and do nothing.
                } else {  // NaN or inf or error.
                    std::string str;
                    is >> str;
                    if (!strcasecmp(str.c_str(), "inf") ||
                        !strcasecmp(str.c_str(), "infinity")) {
                        cur_row->push_back(std::numeric_limits<T>::infinity());
                        LOG_WARN("Reading infinite value into matrix.");
                    } else if (!strcasecmp(str.c_str(), "nan")) {
                        cur_row->push_back(std::numeric_limits<T>::quiet_NaN());
                       LOG_WARN("Reading NaN value into matrix.");
                    } else {
                        LOG_WARN("Expecting numeric matrix data, got " << str);
                        goto cleanup;
                    }
                }
            }
        cleanup: // We only reach here in case of error in the while loop above.
            if(cur_row != NULL)
                delete cur_row;
            for (size_t i = 0; i < data.size(); i++)
                if(data[i] != NULL)
                    delete data[i];
#endif
        }
    bad:
        LOG_ERROR("Failed to read matrix from stream.");
    }
    template<typename T>
    void MatrixBase<T>::Write(std::ostream & os, bool binary) const{
        using MatrixIndexT = int32_t;
        if (!os.good()) {
            LOG_ERROR("Failed to write matrix to stream: stream not good");
        }
        if(binary){ // binary mode
            uint32_t rows = this->rows_;  // make the size 32-bit on disk.
            uint32_t cols = this->cols_;
            WriteUint32(os, rows);
            WriteUint32(os, cols);
#ifdef USE_CUDA
            T* tmp = new T[cols];
            for (MatrixIndexT i = 0; i < (MatrixIndexT)rows; i++) {
                CU_SAFE_CALL(cudaMemcpy(tmp, this->RowData(i), cols*sizeof(T), cudaMemcpyDeviceToHost));
                os.write(reinterpret_cast<const char*> (tmp), sizeof(T) * cols);
            }
            delete []tmp;
#else
            for (MatrixIndexT i = 0; i < rows; i++){
                os.write(reinterpret_cast<const char*> (this->RowData(i)), sizeof(T)
                         * cols);
            }
#endif
            if (!os.good()) {
                LOG_ERROR("Failed to write matrix to stream");
            }
        }else{ // text mode
#ifdef USE_CUDA
            LOG_ASSERT(0 && "Matrix Write() in non-binary format is not supported in GPU");
#else
            if (this->cols_ == 0) {
                os << " [ ]\n";
            } else {
                os << " [";
                for (MatrixIndexT i = 0; i < this->rows_; i++) {
                    os << "\n  ";
                    for (MatrixIndexT j = 0; j < this->cols_; j++)
                        os << (*this)(i, j) << " ";
                }
                os << "]\n";
            }
#endif
        }

    }
        
}// end of namespace DNN
#endif /* nnet_matrix_imp_h */
