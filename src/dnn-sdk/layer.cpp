

#include "layer.h"
#include "tdnn.h"
#include "dense.h"
#include "lstm.h"
#include "rnn.h"
#include "bi-rnn.h"
namespace DNN{
    Layer::Layer(LayerType type):_input_dim(0), _output_dim(0), index(-1), layer_type(type){
        input_context = {0};
    }
    Layer::Layer(int32_t i_dim, int32_t o_dim, LayerType type): layer_type(type){
        LOG_ASSERT(i_dim > 0 && o_dim >0);
        _input_dim = i_dim;
        _output_dim = o_dim;
        index = -1;
        input_context = {0};
    }
    Layer::Layer(int32_t i_dim,
                 int32_t o_dim,
                 const std::vector<int32_t>& context,
                 LayerType type): input_context(context), layer_type(type){
        LOG_ASSERT(i_dim > 0 && o_dim >0);
        _input_dim = i_dim;
        _output_dim = o_dim;
        index = -1;
    }
    Layer::~Layer() = default;
    
    void Layer::Write(std::ostream &os, bool binary) const{
        std::string begin_token = "<" + TypeToMarker(layer_type) + ">";
        std::string end_token = "</" + TypeToMarker(layer_type) + ">";
        WriteToken(os, binary, begin_token);
        WriteBasicType(os, binary, InputDim());
        WriteBasicType(os, binary, OutputDim());
        if (!binary) os << "\n";
        
        WriteData(os, binary);
        
        WriteToken(os, binary, end_token);
        if (!binary) os << "\n";
    }
    Layer* Layer::Read(std::istream &is, bool binary){
        Layer* layer = nullptr;
        std::string token;
        ReadToken(is, binary, &token);
        if (token == "</Nnet>") {
            return nullptr;
        }
        std::string type_name = token;  // name
        std::string type_name_end = type_name;
        type_name_end.insert(type_name_end.begin() + 1 , '/');
        int32_t input_dim, output_dim;
        ReadBasicType(is, binary, &input_dim);
        ReadBasicType(is, binary, &output_dim);
        if(type_name == "<TDNN>"){
            layer = new Tdnn(input_dim, output_dim);
        }else if(type_name == "<DENSE>"){
            layer = new Dense(input_dim, output_dim);
        }else if(type_name == "<RNN>"){
            layer = new RNN(input_dim, output_dim);
        }else if(type_name == "<BRNN>"){
            layer = new BRNN(input_dim, output_dim / 2);
        }else if(type_name == "<LSTM>"){
            layer = new LSTM(input_dim, output_dim);
        }else{
            LOG_ERROR(type_name + " has not implemented yet");
            return layer;
        }
        layer->ReadData(is, binary);
        ExpectToken(is, binary, type_name_end);
        return layer;
    }
    std::string Layer::TypeToMarker(LayerType layer_type) const{
        if(layer_type == LayerType::DENSE){
            return "DENSE";
        }else if(layer_type == LayerType::TDNN){
            return "TDNN";
        }else if(layer_type == LayerType::RNN){
            return "RNN";
        }else if(layer_type == LayerType::BRNN){
            return "BRNN";
        }else if(layer_type == LayerType::LSTM){
            return "LSTM";
        }else{
            // To do ...
            return "";
        }
    }
    void Layer::SetContext(const std::vector<int32_t>& context){
        input_context = context;
    }
    void Layer::SetInputDim(int32_t i_dim){
        _input_dim = i_dim;
    }
    void Layer::SetOutputDim(int32_t o_dim){
        _output_dim = o_dim;
    }
    int32_t Layer::InputDim() const {
        return _input_dim;
    }
    int32_t Layer::OutputDim() const {
        return _output_dim;
    }
    int32_t Layer::GetIndex() const {
        return index;
    }
    void Layer::SetIndex(int32_t i){
        index = i;
    }
    std::vector<int32_t> Layer::Context() const{
        return input_context;
    }
    Layer::LayerType Layer::GetLayerType() const{
        return layer_type;
    }
    void Layer::Propagate(Matrix<float> &in, Matrix<float> *out,
                          std::vector<int>* target_frames) const{
        // Check the dims
        if (_input_dim != in.NumCols() * (input_context.empty() ? 1 : input_context.size())) {
            LOG_ERROR("Non-matching dims on the input of component. The input-dim is " << _input_dim << ", the data had " << in.NumCols() << " dims.");
        }
        // Allocate target buffer
        int diff = 0;
        if(!input_context.empty()) {
            diff = input_context.back() - input_context.front();
        }
        out->Resize(in.NumRows() - diff, _output_dim);  // reset and set it to be 0
        // Call the propagation implementation of the component
        PropagateFnc(&in, out, target_frames);
    }

}// end of namespace DNN
