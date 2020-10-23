

#ifndef layer_h
#define layer_h

#include <stdio.h>
#include "nnet-matrix.h"
#include "nnet-vector.h"
#include "nnet-comm.h"
#include "nnet-math.h"

namespace DNN{
    
    // The abstruct Layer
    class Layer{
    public:
        enum class LayerType{
            TDNN,
            DENSE,
            RNN,
            BRNN,
            LSTM,
            NONE
        };
        Layer(LayerType type = LayerType::NONE);
        Layer(int32_t i_dim, int32_t o_dim, LayerType type = LayerType::NONE);
        Layer(int32_t i_dim, int32_t o_dim, const std::vector<int32_t>& context, LayerType type = LayerType::NONE);
        virtual ~Layer() = 0;
        void Write(std::ostream &os, bool binary) const;
        static Layer* Read(std::istream &is, bool binary);
        virtual void WriteData(std::ostream &os, bool binary) const = 0;
        virtual void ReadData(std::istream &is, bool binary){};
        std::string TypeToMarker(LayerType layer_type) const;
        
        void SetInputDim(int32_t i_dim);
        void SetOutputDim(int32_t o_dim);
        // Get the dimension of the input,
        int32_t InputDim() const;
        // Get the dimension of the output,
        int32_t OutputDim() const;
        int32_t GetIndex() const;
        LayerType GetLayerType() const;
        std::vector<int32_t> Context() const;
        void SetContext(const std::vector<int32_t>& context);
        void SetIndex(int32_t i);
        void Propagate(Matrix<float> &in, Matrix<float> *out,
                       std::vector<int>* target_frames = nullptr) const;

    protected:
        virtual void PropagateFnc(MatrixBase<float> *in, MatrixBase<float> *out,
                                  std::vector<int>* target_frames = nullptr) const = 0;

        int32_t _input_dim = 0;
        int32_t _output_dim = 0;
        int32_t index = -1;
        std::vector<int32_t> input_context;
        LayerType layer_type;
    };

}// end of namespace DNN

#endif /* layer_h */
