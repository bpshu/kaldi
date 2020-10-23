// nnet3bin/nnet3-am-copy.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2016 Daniel Galvez

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-utils.h"
#include "dnn-sdk/nnet.h"
#include "dnn-sdk/tdnn.h"
#include "dnn-sdk/dense.h"
#include "dnn-sdk/activation.h"
using namespace kaldi;
using namespace kaldi::nnet3;
typedef kaldi::int32 int32;

void GetParameters(const CuMatrix<float>& linear,
                   const CuVector<float>& bias,
                   std::vector<std::vector<float>>* W,
                   std::vector<float>* B){
    W->clear();
    B->clear();
    int row = linear.NumRows();
    int col = linear.NumCols();
    int dim = bias.Dim();
    W->resize(row);
    B->resize(dim);
    for(int i = 0; i < row; ++i){
        (*W)[i].resize(col);
        for(int j = 0; j < col; ++j)
            (*W)[i][j] = linear(i ,j);
    }
    for(int i = 0; i < dim; ++i) (*B)[i] = bias(i);
}
int main(int argc, char *argv[]) {
    try {
        const char *usage =
        "From kaldi nnet3 neural-net acoustic model file to transition model and our dnn\n"
        "Usage:  nnet3-conversion [options] <nnet-in> <transition model-out> <nnet-out>\n"
        "e.g.:\n"
        "nnet3-split --binary_write=false --print_nnet_info=false final.mdl tran.mdl final.nnet\n";
        
        bool binary_write = true;
        bool print_nnet_info = false;
        
        ParseOptions po(usage);
        po.Register("binary_write", &binary_write, "Write output in binary mode");
        po.Register("print_nnet_info", &print_nnet_info, "Write output in binary mode");

        po.Read(argc, argv);
        
        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }
        std::string nnet_rxfilename = po.GetArg(1);
        std::string tran_wxfilename = po.GetArg(2);
        std::string nnet_wxfilename = po.GetArg(3);
        
        TransitionModel trans_model;
        AmNnetSimple am_nnet;
        {
            bool binary;
            Input ki(nnet_rxfilename, &binary);
            trans_model.Read(ki.Stream(), binary);
            am_nnet.Read(ki.Stream(), binary);
        }
        if(print_nnet_info)
            KALDI_LOG << am_nnet.Info();
        // {
        //     bool b = false;
        //     Output ki2("./tmp/final.mdl.txt", &b);
        //     am_nnet.Write(ki2.Stream(), b);    
        // }

        Nnet nnet = am_nnet.GetNnet();
        const VectorBase<BaseFloat>& prior = am_nnet.Priors();
        SetBatchnormTestMode(true, &nnet);
        SetDropoutTestMode(true, &nnet);
        CollapseModel(CollapseModelConfig(), &nnet);
        DNN::Nnet new_nnet;
        std::vector<float> p;
        p.resize(prior.Dim());
        for(int i = 0; i < prior.Dim(); ++i){
            p[i] = prior(i);
        }
        new_nnet.SetPrior(p);
        int ncomponent = nnet.NumComponents();
        std::string name = nnet.GetComponentName(ncomponent-1);
        assert(name == "lda.tdnn1.affine");
        std::vector<std::vector<float>> W;
        std::vector<float> B;
        AffineComponent* affine = dynamic_cast<AffineComponent*>(nnet.GetComponent(ncomponent-1));
        GetParameters(affine->LinearParams(), affine->BiasParams(), &W, &B);
        new_nnet.AddLayer(new DNN::Tdnn(W, B, {-2,-1,0,1,2}, DNN::ACTIVATION_TYPE::RELU, true)); // tdnn1
        //first get the last component
        for(int i = 0; i < ncomponent - 1; ++i){
            name = nnet.GetComponentName(i);
            if(name.find(".affine") != std::string::npos){
                affine = dynamic_cast<AffineComponent*>(nnet.GetComponent(i));
                GetParameters(affine->LinearParams(), affine->BiasParams(), &W, &B);
                if(name == "tdnn2.affine"){
                    new_nnet.AddLayer(new DNN::Tdnn(W, B, {-1, 0, 1}, DNN::ACTIVATION_TYPE::RELU, true)); //tdnn2
                }
                if(name == "tdnn3.affine"){
                    new_nnet.AddLayer(new DNN::Tdnn(W, B, {-3, 0, 3}, DNN::ACTIVATION_TYPE::RELU, true)); // tdnn3
                }
                if(name == "tdnn4.affine"){
                    new_nnet.AddLayer(new DNN::Tdnn(W, B, {-3, 0, 3}, DNN::ACTIVATION_TYPE::RELU, true)); // tdnn3
                }
                // if(name == "tdnn5.affine"){
                //     new_nnet.AddLayer(new DNN::Tdnn(W, B, {-3, 0, 3}, DNN::ACTIVATION_TYPE::RELU, true)); // tdnn3
                // }
           		//if(name == "tdnn6.affine"){
                //    new_nnet.AddLayer(new DNN::Tdnn(W, B, {-3, 0, 3}, DNN::ACTIVATION_TYPE::RELU, true)); // tdnn4
                //}
                if(name == "tdnn5.affine"){
                    new_nnet.AddLayer(new DNN::Tdnn(W, B, {-6, -3, 0}, DNN::ACTIVATION_TYPE::RELU, true)); // tdnn5
                }
                if(name == "prefinal-chain.affine"){
                    new_nnet.AddLayer(new DNN::Tdnn(W, B, {0}, DNN::ACTIVATION_TYPE::RELU, true)); // tdnn_prefinal_chain
                }
                if(name == "output.affine"){ 
                    new_nnet.AddLayer(new DNN::Dense(W, B, DNN::ACTIVATION_TYPE::NONE));
                } // output layer
            }
        }
        WriteKaldiObject(trans_model, tran_wxfilename, binary_write);
        std::ofstream out_nnet(nnet_wxfilename);
        if(out_nnet.good()){
            new_nnet.Write(out_nnet, binary_write);
        }
	    KALDI_LOG << new_nnet.NnetInfo();
        out_nnet.close();
        DNN::Nnet xxx;
        std::ifstream if_nnet(nnet_wxfilename);
        xxx.Read(if_nnet, binary_write);
        if_nnet.close();
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return -1;
    }
}
