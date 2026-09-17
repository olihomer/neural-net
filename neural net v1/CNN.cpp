//
//  CNN.cpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#include "CNN.hpp"
#include <span>
#include <iostream>
#include <chrono>

namespace
{
    constexpr bool profileCNN = false;
}

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

void CNN::backward(const Tensor& outputGradient)
{
    std::chrono::steady_clock::time_point conv2BackwardStart;
    if constexpr (profileCNN)
        conv2BackwardStart = std::chrono::steady_clock::now();

    Tensor x2 = conv2_.backward(outputGradient, true);

    std::chrono::duration<double> conv2BackwardElapsed{0.0};
    if constexpr (profileCNN)
        conv2BackwardElapsed = std::chrono::steady_clock::now() - conv2BackwardStart;

    std::chrono::steady_clock::time_point conv1BackwardStart;
    if constexpr (profileCNN)
        conv1BackwardStart = std::chrono::steady_clock::now();

    conv1_.backward(x2, false);

    std::chrono::duration<double> conv1BackwardElapsed{0.0};
    if constexpr (profileCNN)
        conv1BackwardElapsed = std::chrono::steady_clock::now() - conv1BackwardStart;

    if constexpr (profileCNN)
    {
        const double totalBackwardSeconds = conv2BackwardElapsed.count() + conv1BackwardElapsed.count();

        if(totalBackwardSeconds > 0.0)
        {
            std::cout << "CNN::backward layer benchmark:" << std::endl;
            std::cout << "  conv2_.backward: " << conv2BackwardElapsed.count() << " seconds, "
                      << (conv2BackwardElapsed.count() / totalBackwardSeconds) * 100.0 << "%" << std::endl;
            std::cout << "  conv1_.backward: " << conv1BackwardElapsed.count() << " seconds, "
                      << (conv1BackwardElapsed.count() / totalBackwardSeconds) * 100.0 << "%" << std::endl;
        }
    }
    
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
    std::chrono::steady_clock::time_point batchStart;
    if constexpr (profileCNN)
        batchStart = std::chrono::steady_clock::now();

    std::chrono::duration<double> forwardElapsed{0.0};
    std::chrono::duration<double> classifierTrainElapsed{0.0};
    std::chrono::duration<double> backwardElapsed{0.0};
    
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
        std::chrono::steady_clock::time_point forwardStart;
        if constexpr (profileCNN)
            forwardStart = std::chrono::steady_clock::now();

        auto outputVector = forward(input);
        
        //cache network values for backwards pass
        conv1_.pushCache();
        conv2_.pushCache();

        if constexpr (profileCNN)
            forwardElapsed += std::chrono::steady_clock::now() - forwardStart;
        
        for(std::size_t j=0; j<outputVector.size(); j++)
            MLPinputs(j,index)=outputVector[j];
    }

    //Batch backprop through MLP
    
    std::chrono::steady_clock::time_point classifierTrainStart;
    if constexpr (profileCNN)
        classifierTrainStart = std::chrono::steady_clock::now();

    total_error = classifier_.trainBatch(MLPinputs, MLPtargets);

    if constexpr (profileCNN)
        classifierTrainElapsed += std::chrono::steady_clock::now() - classifierTrainStart;

    Matrix inputError = classifier_.get_input_error();

    //inputError matrix now contains error gradients for entire batch. Backprop through CNN one at a time.
    
    Tensor outputGradient({16,7,7});
 
    for(std::size_t index=0; index<batch.size(); index++)
    {
        //retrieve cache values from forward run in order for backprop to work
        
        std::chrono::steady_clock::time_point forwardStart;
        if constexpr (profileCNN)
            forwardStart = std::chrono::steady_clock::now();

        conv1_.popCache();
        conv2_.popCache();

        if constexpr (profileCNN)
            forwardElapsed += std::chrono::steady_clock::now() - forwardStart;
        
        for(std::size_t i=0;i<outputGradient.size();i++)outputGradient.data()[i] = inputError(i,index);

        std::chrono::steady_clock::time_point backwardStart;
        if constexpr (profileCNN)
            backwardStart = std::chrono::steady_clock::now();

        backward(outputGradient);

        if constexpr (profileCNN)
            backwardElapsed += std::chrono::steady_clock::now() - backwardStart;
    }

    if constexpr (profileCNN)
    {
        const auto batchEnd = std::chrono::steady_clock::now();
        const std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
        const double batchSeconds = batchElapsed.count();
        
        if(batchSeconds > 0.0)
        {
            std::cout << "CNN trainBatch benchmark: " << batch.size() << " samples in "
                      << batchSeconds << " seconds" << std::endl;
            std::cout << "  CNN::forward: " << forwardElapsed.count() << " seconds, "
                      << (forwardElapsed.count() / batchSeconds) * 100.0 << "%" << std::endl;
            std::cout << "  Neural::trainBatch: " << classifierTrainElapsed.count() << " seconds, "
                      << (classifierTrainElapsed.count() / batchSeconds) * 100.0 << "%" << std::endl;
            std::cout << "  CNN::backward: " << backwardElapsed.count() << " seconds, "
                      << (backwardElapsed.count() / batchSeconds) * 100.0 << "%" << std::endl;
        }
    }
    
    return total_error;
}

void CNN::print_stats(std::ostream& ostream)
{
    ;
}
