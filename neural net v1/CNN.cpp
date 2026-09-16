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
    
    /*classifier_.set_input(flat);
    classifier_.propagate();
 
    return classifier_.get_output();*/
    
    return flat;
}

Tensor CNN::backward(const Tensor& outputGradient)
{
    // run classifier backwards
    // unflatten
    
    auto x2 = conv1_.backward(outputGradient);
    auto x1 = conv1_.backward(x2);
    
    return x1;
}


void CNN::gradient_descent(std::size_t trainingSize, double learningRate)
{
    ;
}

double CNN::trainBatch(const data_set& training_data, const std::span<const std::size_t> batch)
{
    std::cout << "Start of trainBatch" << std::endl;

    auto data = training_data.get_data()[0].inputs;
    Tensor input({1,28,28});
    
    for(std::size_t y = 0; y < 28; y++)
    {
        for(std::size_t x = 0; x < 28; x++)
        {
            input(0,y,x) = data[x+y*28];
            std::cout << (data[x+y*28] > 50.0f/255.0f ? "X " : "  ");
        }
        std::cout << std::endl;
    }
    
    //Put through CNN
    
    auto x = forward(input);
    
    data_set CNNresult{x,training_data.get_data()[0].outputs};
    
    std::vector<size_t> singlebatch = {0};
    
    //Put through MLP
    classifier_.trainBatch(CNNresult, std::span<std::size_t>(singlebatch.begin(),1));

    Tensor outputGradient({16,7,7});
    auto inputError = classifier_.get_input_error();
    
    for(std::size_t i=0;i<outputGradient.size();i++)outputGradient.data()[i] = inputError[i];
    
    backward(outputGradient);
    
    return 0;
}

void CNN::print_stats(std::ostream& ostream)
{
    ;
}
