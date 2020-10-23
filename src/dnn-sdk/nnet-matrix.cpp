

#include "nnet-matrix.h"

namespace DNN{
    std::atomic<int> MatrixBaseBase::skip(1);
    int MatrixBaseBase::Skip(){
        return skip.load();
    }
    void MatrixBaseBase::SetSkip(int i){
        skip.store(i);
    }
}
