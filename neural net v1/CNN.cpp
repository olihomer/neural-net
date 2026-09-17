//
//  CNN.cpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#include "CNN.hpp"
#include <span>
#include <iostream>



CNN::CNN()
:conv1_(8,1,28,28), conv2_(16,8,14,14),classifier_({784,128,10}, ActivationType::Relu, ActivationType::Softmax)
{
    ;
}

void CNN::print() const
{
    ;
}


std::vector<Scalar> CNN::forward(const Tensor& input)
{
    auto& x1 = conv1_.forward(input);
    auto& x2 = conv2_.forward(x1);
    
    auto flat = flatten_(x2);
    
    return flat;
}

Tensor CNN::backward(const Tensor& outputGradient)
{
    auto x2 = conv2_.backward(outputGradient);
    auto x1 = conv1_.backward(x2);
    
    return x1;
}


void CNN::gradient_descent(std::size_t trainingSize, double learningRate)
{
    const Scalar scale = static_cast<Scalar>(learningRate) / static_cast<Scalar>(trainingSize);
    
    conv1_.gradient_descent(scale);
    conv2_.gradient_descent(scale);
    classifier_.gradient_descent(trainingSize, learningRate);
}

double CNN::trainBatch(const data_set& training_data, const std::span<const std::size_t> batch)
{
    double total_error = 0;
    
    conv1_.zeroGradients();
    conv2_.zeroGradients();
    
    Matrix MLPinputs(classifier_.nInputs_(),batch.size());
    Matrix MLPtargets(classifier_.nOutputs_(),batch.size());
    
    //CNN forward pass and populate matrix of CNN outputs for entire batch
    for(std::size_t index=0; index<batch.size(); index++)
    {
        //load example into Tensor
        Tensor input({1,28,28});
        
        for(std::size_t i=0; i<training_data.n_inputs(); i++)
            input.data()[i]=training_data.get_data()[batch[index]].inputs[i];
        
        for(std::size_t i=0; i<training_data.n_outputs(); i++)
            MLPtargets(i,index)=training_data.get_data()[batch[index]].outputs[i];
        
        //Put through CNN
        auto outputVector = forward(input);
        
        for(std::size_t j=0; j<outputVector.size(); j++)
            MLPinputs(j,index)=outputVector[j];
    }

    //Batch backprop through MLP
    
    total_error = classifier_.trainBatch(MLPinputs, MLPtargets);
    Matrix inputError = classifier_.get_input_error();

    //inputError matrix now contains error gradients for entire batch. Backprop through CNN one at a time.
    
    Tensor outputGradient({16,7,7});
 
    for(std::size_t index=0; index<batch.size(); index++)
    {
        //need to re-run each example forward to get the right activations for backprop to work
        
        Tensor input({1,28,28});
        
        for(std::size_t i=0; i<training_data.n_inputs(); i++)
            input.data()[i]=training_data.get_data()[batch[index]].inputs[i];
        
        forward(input);
        
        for(std::size_t i=0;i<outputGradient.size();i++)outputGradient.data()[i] = inputError(i,index);
        backward(outputGradient);
    }
        
    return total_error;
}

void CNN::print_stats(std::ostream& ostream)
{
    ;
}
