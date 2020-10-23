

#ifndef nnet_vector_imp_h
#define nnet_vector_imp_h

namespace DNN{
    template<typename T>
    Vector<T>::Vector(){
        dim_ = 0;
        stride_ = 0;
        data_ = 0;
    }
    template<typename T>
    Vector<T>::Vector(const std::vector<T>& V){
        this->ReadFromVec(V);
    }
    template<typename T>
    Vector<T>::Vector(const Vector<T> & V){
        this->CopyFromVec(V);
    }
    template<typename T>
    Vector<T>::Vector(const MatrixBase<T>& Mat, int row){
        this->Resize(Mat.NumCols());
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMemcpy(this->Data(), Mat.RowData(row), sizeof(T)*Mat.Stride(), cudaMemcpyDeviceToDevice));
#else
        memcpy(this->Data(), Mat.RowData(row), sizeof(T)*Mat.Stride());
#endif
    }
    template<typename T>
    Vector<T>& Vector<T>::operator =(const Vector<T> & V){
        this->CopyFromVec(V);
        return *this;
    }
    template<typename T>
    void Vector<T>::Release(){
        if(data_)  {
            aligned_free(data_);
            data_ = nullptr;
        }
        dim_ = 0;
        stride_ = 0;
        return;
    }
    template<typename T>
    Vector<T>::~Vector(){
        this->Release();
    }
    template<typename T>
    void Vector<T>::SetZero(){
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMemset(data_, 0, stride_*sizeof(T)));
#else
        if(data_)
            memset(data_, 0, sizeof(T) * stride_);
#endif
    }
    template<typename T>
    void Vector<T>::Resize(int dim){
        this->Release();
        if(dim == 0){
            return;
        }
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMalloc((void**)&data_, dim * sizeof(T)));
        this->dim_ = dim;
        this->stride_ = dim;
#else
        int skip = ((ALIGN / sizeof(T)) - dim % (ALIGN / sizeof(T))) % (ALIGN / sizeof(T));
        stride_ = dim  + skip;
        dim_ = dim;
        data_ = AlignedAlloc<T>(stride_);
        if (!data_){
            throw std::bad_alloc();
        }
#endif
        this->SetZero();
    }
    template<typename T>
    void Vector<T>::CopyFromVec(const Vector<T> &other){
        this->Release();
        this->Resize(other.Dim());
        if (data_ != other.data_) {
#ifdef USE_CUDA
            CU_SAFE_CALL(cudaMemcpy(this->data_, other.data_, stride_*sizeof(T), cudaMemcpyHostToDevice));
#else
            memcpy(this->data_, other.data_, stride_ * sizeof(T));
#endif
         }
    }
    template<typename T>
    void Vector<T>::ReadFromVec(const std::vector<T> &V){
        this->Resize(V.size());
#ifdef USE_CUDA
        CU_SAFE_CALL(cudaMemcpy(this->data_, V.data(), stride_*sizeof(T), cudaMemcpyHostToDevice));
#else
        for(int i = 0; i < Dim(); ++i){
            (*this)(i) = V[i];
        }
#endif
    }
    template<typename T>
    void Vector<T>::Write(std::ostream & os, bool binary) const{
        using MatrixIndexT = int32_t;
        if (!os.good()) {
            LOG_ERROR ("Failed to write vector to stream: stream not good");
        }
        if (binary) {
            uint32_t size = Dim();  // make the size 32-bit on disk.
            WriteUint32(os, size);
#ifdef USE_CUDA
            T* tmp = new T[size];
            CU_SAFE_CALL(cudaMemcpy(tmp, data_, size*sizeof(T), cudaMemcpyDeviceToHost));
            os.write(reinterpret_cast<const char*>(tmp), sizeof(T) * size);
            delete [] tmp;
#endif
            os.write(reinterpret_cast<const char*>(Data()), sizeof(T) * size);
            if (os.fail()) {
                std::cerr << "Error reading vector data (binary mode); truncated "
                "stream? (size = " << size << ")" << std::endl;
                exit(-1);
            }
        } else {
#ifdef USE_CUDA
            LOG_ASSERT("Vector Write() in non-binary format is not supported in GPU");
#else
            os << " [ ";
            for (MatrixIndexT i = 0; i < Dim(); i++){
                os << (*this)(i) << " ";
            }
            os << "]\n";
#endif
        }
        if (!os.good())
            LOG_ERROR("Failed to write vector to stream");
    }
    
    template<typename T>
    void Vector<T>::Read(std::istream & is, bool binary){
        
        using MatrixIndexT = int32_t;
        
        std::ostringstream specific_error;
        MatrixIndexT pos_at_start = is.tellg();
        if (binary) {
            uint32_t size = ReadUint32(is);
            if ((MatrixIndexT)size != this->Dim())  this->Resize(size);
            if (size > 0){
#ifdef USE_CUDA
                T* tmp = new T[size];
                is.read(reinterpret_cast<char*>(tmp), sizeof(T)*size);
                CU_SAFE_CALL(cudaMemcpy(data_, tmp, size*sizeof(T), cudaMemcpyHostToDevice));
                delete [] tmp;
#else
                is.read(reinterpret_cast<char*>(this->data_), sizeof(T)*size);
#endif
            }
            if (is.fail()) {
                specific_error << "Error reading vector data (binary mode); truncated "
                "stream? (size = " << size << ")";
//                goto bad;
            }
            return;
        } else {  // Text mode reading; format is " [ 1.1 2.0 3.4 ]\n"
#ifdef USE_CUDA
            LOG_ASSERT("Vector Read() in non-binary format is not supported in GPU");
#else
            std::string s;
            is >> s;
            // if ((s.compare("DV") == 0) || (s.compare("FV") == 0)) {  // Back compatibility.
            //  is >> s;  // get dimension
            //  is >> s;  // get "["
            // }
            if (is.fail()) { specific_error << "EOF while trying to read vector."; goto bad; }
            if (s.compare("[]") == 0) { Resize(0); return; } // tolerate this variant.
            if (s.compare("[")) { specific_error << "Expected \"[\" but got " << s; goto bad; }
            std::vector<T> data;
            while (1) {
                int i = is.peek();
                if (i == '-' || (i >= '0' && i <= '9')) {  // common cases first.
                    T r;
                    is >> r;
                    if (is.fail()) { specific_error << "Failed to read number."; goto bad; }
                    if (! std::isspace(is.peek()) && is.peek() != ']') {
                        specific_error << "Expected whitespace after number."; goto bad;
                    }
                    data.push_back(r);
                    // But don't eat whitespace... we want to check that it's not newlines
                    // which would be valid only for a matrix.
                } else if (i == ' ' || i == '\t') {
                    is.get();
                } else if (i == ']') {
                    is.get();  // eat the ']'
                    this->Resize(data.size());
                    for (size_t j = 0; j < data.size(); j++)
                        this->data_[j] = data[j];
                    i = is.peek();
                    if (static_cast<char>(i) == '\r') {
                        is.get();
                        is.get();  // get \r\n (must eat what we wrote)
                    } else if (static_cast<char>(i) == '\n') { is.get(); } // get \n (must eat what we wrote)
                    if (is.fail()) {
                        LOG_WARN("After end of vector data, read error.");
                        // we got the data we needed, so just warn for this error.
                    }
                    return;  // success.
                } else if (i == -1) {
                    specific_error << "EOF while reading vector data.";
                    goto bad;
                } else if (i == '\n' || i == '\r') {
                    specific_error << "Newline found while reading vector (maybe it's a matrix?)";
                    goto bad;
                } else {
                    is >> s;  // read string.
                    if (!strcasecmp(s.c_str(), "inf") ||
                        !strcasecmp(s.c_str(), "infinity")) {
                        data.push_back(std::numeric_limits<T>::infinity());
                        LOG_WARN("Reading infinite value into vector.");
                    } else if (!strcasecmp(s.c_str(), "nan")) {
                        data.push_back(std::numeric_limits<T>::quiet_NaN());
                        LOG_WARN("Reading NaN value into vector.");
                    } else {
                        specific_error << "Expecting numeric vector data, got " << s;
                        goto  bad;
                    }
                }
            }
#endif
        }
        // we never reach this line (the while loop returns directly).
    bad:
        LOG_ERROR("Failed to read vector from stream.  " << specific_error.str()
                  << " File position at start is "
                  << pos_at_start<<", currently "<<is.tellg());
    }
    
} // end of namespace DNN

#endif /* nnet_vector_imp_h */
