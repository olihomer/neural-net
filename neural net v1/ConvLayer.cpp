//
//  ConvLayer.cpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#include "ConvLayer.hpp"
#include <cstddef>
#include <stdio.h>
#include "Tensor.hpp"
#include "NeuralTypes.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

ConvLayer::ConvLayer(std::size_t outputChannels,
                     std::size_t inputChannels,
                     std::size_t inputHeight,
                     std::size_t inputWidth)
{
    input_ = Tensor({inputChannels,inputHeight,inputWidth});
    activation_ = Tensor({outputChannels,inputHeight,inputWidth});
    pooled_ = Tensor({outputChannels,inputHeight/stride,inputWidth/stride});
    maxPoolSource_ = Tensor({outputChannels,inputHeight/stride,inputWidth/stride});
    kernels_ = Tensor({outputChannels,inputChannels,3,3});
    biases_.resize(outputChannels);
}

const Tensor& ConvLayer::forward (const Tensor& input)
{
    input_ = input;
    convolve_();
    maxPool_();
    return pooled_;
}


void ConvLayer::convolve_()
{
    auto inputChannels = input_.dim(0);
    auto inputY = input_.dim(1);
    auto inputX = input_.dim(2);
    auto outputChannels = kernels_.dim(0);
    
    //same-padding with integer loops to handle edges more easily
    for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
        for(int j = 0; j < inputY; j++)
            for(int i = 0; i < inputX; i++)
            {
                Scalar sum = 0;
                for(int n=-1;n<2;n++)
                    for(int m=-1;m<2;m++)
                        for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                        {
                            sum += (j+n<0 || j+n>inputY-1 || i+m<0 || i+m>inputX-1) ? 0.0f : input_(0,j+n,i+m) * kernels_(outChan,inChan,n+1,m+1);
                        }
                activation_(outChan,j,i) = (biases_[outChan]+sum) > 0.0 ? (biases_[outChan]+sum) : 0.0;
            }
    
}


void ConvLayer::maxPool_()
{
    std::size_t channels = activation_.dim(0);
    
    std::size_t outputY = pooled_.dim(1);
    std::size_t outputX = pooled_.dim(2);
    
    std::vector<Scalar> candidates;
    candidates.resize(stride*stride);
    
    for(std::size_t chan = 0; chan < channels; chan++)
    {
        std::size_t indexX = 0;
        std::size_t indexY = 0;
        
        for(std::size_t j=0;j<outputY;j++)
        {
            for(std::size_t i=0;i<outputX;i++)
            {
                for(std::size_t x=0;x<stride;x++)
                    for(std::size_t y=0;y<stride;y++)
                    {
                        candidates[x+y*stride] = activation_(chan,indexY + y,indexX + x);
                    }
                auto max = std::max_element(candidates.begin(), candidates.end());
                pooled_(chan,j,i) = *max;
                maxPoolSource_(chan,j,i) = max - candidates.begin();
                indexX += stride;
            }
            indexY += stride;
            indexX = 0;
        }
    }
}

void ConvLayer::setKernel(std::size_t outputChannel, std::size_t inputChannel, const std::vector<Scalar>& data)
{
        auto d = data.begin();
        for(std::size_t j=0; j<kernels_.dim(2); j++)
            for(std::size_t i=0; i<kernels_.dim(3); i++)
                kernels_(outputChannel,inputChannel,j,i) = *(d++);
}


void ConvLayer::print() const
{
    kernels_.print();
    
    std::cout << "Kernel data:" << std::endl;
    
    for(std::size_t l=0; l<kernels_.dim(0); l++)
    {
        std::cout << "Output Channel " << l << std::endl;
        for(std::size_t j=0; j<kernels_.dim(1); j++)
        {
            for(std::size_t i=0; i<kernels_.dim(2); i++)
                std::cout << kernels_(l,0,j,i) << " ";
            std::cout << std::endl;
        }
    }
    
    std::cout << "Activation data:" << std::endl;
    
    for(std::size_t l=0; l<activation_.dim(0); l++)
    {
        std::cout << "Channel " << l << std::endl;
        for(std::size_t j=0; j<activation_.dim(1); j++)
        {
            for(std::size_t i=0; i<activation_.dim(2); i++)
                std::cout << activation_(l,j,i) << " ";
            std::cout << std::endl;
        }
    }
    
    std::cout << "MaxPool data:" << std::endl;
    
    for(std::size_t l=0; l<pooled_.dim(0); l++)
    {
        std::cout << "Channel " << l << std::endl;
        for(std::size_t j=0; j<pooled_.dim(1); j++)
        {
            for(std::size_t i=0; i<pooled_.dim(2); i++)
                std::cout << pooled_(l,j,i) << " ";
            std::cout << std::endl;
        }
    }
    
}
