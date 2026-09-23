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
#include <chrono>
#include <array>
#include <random>


std::random_device ConvLayer::rd_;
std::mt19937 ConvLayer::rng_(ConvLayer::rd_());

namespace
{
constexpr bool profileConvLayer = false;
}

ConvLayer::ConvLayer(std::size_t outputChannels,
                     std::size_t inputChannels,
                     std::size_t inputHeight,
                     std::size_t inputWidth)
:input_({inputChannels,inputHeight,inputWidth}),
activation_({outputChannels,inputHeight,inputWidth}),
pooled_({outputChannels,inputHeight/stride,inputWidth/stride}),
maxPoolSource_({outputChannels,inputHeight/stride,inputWidth/stride}),
kernels_({outputChannels,inputChannels,3,3}),
kernelGradient_({outputChannels,inputChannels,3,3})
{
    biases_.resize(outputChannels);
    biasGradient_.resize(outputChannels);
    
    initialiseWeights();
}

const Tensor& ConvLayer::forward (const Tensor& input)
{
    input_ = input;

    convolve_();

    maxPool_();

    return pooled_;
}

Scalar ConvLayer::kernelValue(std::size_t outputChannel, std::size_t inputChannel, std::size_t y, std::size_t x) const
{
    return kernels_(outputChannel, inputChannel, y, x);
}

Scalar ConvLayer::inputValue(std::size_t channel, std::size_t y, std::size_t x) const
{
    return input_(channel, y, x);
}

Scalar ConvLayer::activationValue(std::size_t channel, std::size_t y, std::size_t x) const
{
    return activation_(channel, y, x);
}

Tensor ConvLayer::backward(const Tensor& outputGradient, const bool returnInputGradient)
{
    //unpool
    //Relu derivative
    //unconvolve
    // - kernel gradients
    // - bias gradients
    // - input gradients
    // return input gradients
    
    if(outputGradient.shape()!=pooled_.shape())
        throw std::runtime_error("Backwards gradient shape mismatch");
    
    // unpool + unRelu
    
    const std::size_t outputChannels = activation_.dim(0);
    const std::size_t outputY = outputGradient.dim(1);
    const std::size_t outputX = outputGradient.dim(2);
    
    //declaration of variables/pointers that might not get used
    Tensor inputGradient;
    Scalar* inputGradientData;
    Scalar* inputGradientInChannel;
    Scalar* inputGradientRow;
    
    if(returnInputGradient==true)
    {
        inputGradient = Tensor({input_.dim(0),input_.dim(1),input_.dim(2)});
        Scalar* inputGradientData = inputGradient.data();
    }
        
    const auto inputY = input_.dim(1);
    const auto inputX = input_.dim(2);
    const auto inputChannels = input_.dim(0);

    //faster access code
    
    const Scalar* kernelData = kernels_.data();
    Scalar* kernelGradientData = kernelGradient_.data();
    const Scalar* inputData = input_.data();
    const Scalar* activationData = activation_.data();
    const Scalar* maxPoolSourceData = maxPoolSource_.data();
    const Scalar* outputGradientData = outputGradient.data();
    
    const std::size_t kernelStride = kernels_.dim(2) * kernels_.dim(3);
    const std::size_t kernelX = kernels_.dim(3);
    
    for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
    {
        std::size_t indexX = 0;
        std::size_t indexY = 0;
        
        const Scalar* kernelOutChannel = kernelData + (outChan * inputChannels * kernelStride);
        Scalar* kernelGradientOutChannel = kernelGradientData + (outChan * inputChannels * kernelStride);
        const Scalar* activationChannel = activationData + (outChan * inputY * inputX);
        const Scalar* maxPoolSourceChannel = maxPoolSourceData + (outChan * outputY * outputX);
        const Scalar* outputGradientChannel = outputGradientData + (outChan * outputY * outputX);
        
        bool nodanger = false;
        
        for(std::size_t j=0;j<outputY;j++)
        {
            const Scalar* maxPoolSourceRow = maxPoolSourceChannel + (j * outputX);
            const Scalar* outputGradientRow = outputGradientChannel + (j * outputX);

            for(std::size_t i=0;i<outputX;i++)
            {
                //identify winner from activation tensor
                std::size_t index = (std::size_t)(*(maxPoolSourceRow + i));
                std::size_t dx = (index == 2 || index == 0) ? 0 : 1;
                std::size_t dy = index < 2 ? 0 : 1;
                
                std::size_t winX = indexX + dx;
                std::size_t winY = indexY + dy;
                
                nodanger = false;
                if(winX > 0 && winX < inputX-2 && winY > 0 && winY < inputY-2)nodanger=true;
                
                if((*(activationChannel + (winY * inputX) + winX) > 0.0)) //unRelu => only carry gradient back if positive activation
                {
                    //carry back the gradient. store as a local temp for now to avoid multiple lookups
                    const auto g = *(outputGradientRow + i);
                    
                    // do the unconvolve here!
                    // (winY, winX) is a coordinate of a live preactivation
                   
                    for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                    {
                        Scalar* kernelGradientInChannel = kernelGradientOutChannel + (inChan * kernelStride);
                        const Scalar* kernelInChannel = kernelOutChannel + (inChan * kernelStride);
                        const Scalar* inputInChannel = inputData + (inChan * inputY * inputX);
                        if(returnInputGradient==true)
                            inputGradientInChannel = inputGradientData + (inChan * inputY * inputX);
                        
                        for(int n=-1;n<2;n++)
                        {
                            Scalar* kernelGradientRow = kernelGradientInChannel + ((n + 1) * kernelX);
                            const Scalar* kernelRow = kernelInChannel + + ((n + 1) * kernelX);
                            const Scalar* inputRow = inputInChannel + ((winY + n) * inputX);
                            if(returnInputGradient==true)
                                inputGradientRow = inputGradientInChannel + ((winY + n) * inputX);
                            
                            for(int m=-1;m<2;m++)
                            {
                                if(nodanger)
                                {
                                    (*(kernelGradientRow + (m + 1))) += (*(inputRow + (winX + m))) * g;
                                    if(returnInputGradient==true)
                                        (*(inputGradientRow + (winX + m))) += (*(kernelRow + (m + 1))) * g;
                                }
                                else
                                {
                                    if(!(winY+n<0 || winY+n>inputY-1 || winX+m<0 || winX+m>inputX-1))
                                    {
                                        (*(kernelGradientRow + (m + 1))) += (*(inputRow + (winX + m))) * g;
                                        if(returnInputGradient==true)
                                            (*(inputGradientRow + (winX + m))) += (*(kernelRow + (m + 1))) * g;
                                    }
                                }
                            }
                        }
                        biasGradient_[outChan] += g;
                    }
                }
                
                indexX += stride;
            }
            indexY += stride;
            indexX = 0;
        }
    }
    
    return inputGradient;
    
}


void ConvLayer::convolve_()
{
    const auto inputChannels = input_.dim(0);
    const auto inputY = input_.dim(1);
    const auto inputX = input_.dim(2);
    const auto outputChannels = kernels_.dim(0);
    
    //faster access code
    
    const Scalar* inputData = input_.data();
    const Scalar* kernelData = kernels_.data();
    Scalar* activationData = activation_.data();
    const std::size_t channelStride = inputY * inputX;
    const std::size_t kernelStride = kernels_.dim(2) * kernels_.dim(3);
    const std::size_t kernelY = kernels_.dim(2);
    bool nodanger = false;
    
    //same-padding with integer loops to handle edges more easily
    for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
    {
        const Scalar* kernelOutChannel = kernelData + outChan * kernelStride * inputChannels;
        Scalar* activationChannel = activationData + outChan * channelStride;
        for(int j = 0; j < inputY; j++)
        {
            Scalar* activationRow = activationChannel + (j * inputX);
            for(int i = 0; i < inputX; i++)
            {
                nodanger = false;
                if(j > 0 && i > 0 && j < inputY-2 && i < inputX-2)nodanger = true;
                Scalar sum = biases_[outChan];
                for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                {
                    const Scalar* channel = inputData + inChan * channelStride;
                    const Scalar* kernelInputChannel = kernelOutChannel + inChan * kernelStride;
                    
                    if(nodanger)
                    {
                        const Scalar* r0 = channel + (j - 1) * inputX + i - 1;
                        const Scalar* r1 = channel + j * inputX + i - 1;
                        const Scalar* r2 = channel + (j + 1) * inputX + i - 1;
                        
                        sum += r0[0] * kernelInputChannel[0]
                            + r0[1] * kernelInputChannel[1]
                            + r0[2] * kernelInputChannel[2]
                            + r1[0] * kernelInputChannel[3]
                            + r1[1] * kernelInputChannel[4]
                            + r1[2] * kernelInputChannel[5]
                            + r2[0] * kernelInputChannel[6]
                            + r2[1] * kernelInputChannel[7]
                            + r2[2] * kernelInputChannel[8];
                    }
                    else
                    {
                        for(int n=-1;n<2;n++)
                        {
                            const Scalar* row = channel + (j+n) * inputX + i;
                            const Scalar* kernelRow = kernelInputChannel + (n+1) * kernelY;
                            
                            for(int m=-1;m<2;m++)
                            {
                                sum += (j+n<0 || j+n>inputY-1 || i+m<0 || i+m>inputX-1) ? 0.0f : (*(row + m)) * (*(kernelRow + m + 1));
                            }
                        }
                    }
                }
                *(activationRow + i) = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
}

void ConvLayer::gradient_descent(const Scalar scale)
{
    const auto outputChannels = kernels_.dim(0);
    const auto inputChannels = kernels_.dim(1);
    const auto kY = kernels_.dim(2);
    const auto kX = kernels_.dim(3);
    const auto kernelStride = kY * kX;
   
    //faster access code
    Scalar* kernelData = kernels_.data();
    const Scalar* kernelGradientData = kernelGradient_.data();
    
        for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
        {
            Scalar* kernelOutChannel = kernelData + (outChan * inputChannels * kernelStride);
            const Scalar* kernelGradientOutChannel = kernelGradientData + (outChan * inputChannels * kernelStride);
            
            for(std::size_t inChan = 0; inChan < inputChannels; inChan++)
            {
                Scalar* kernelInChannel = kernelOutChannel + (inChan * kernelStride);
                const Scalar* kernelGradientInChannel = kernelGradientOutChannel + (inChan * kernelStride);
                
                for(int j = 0; j < kY; j++)
                {
                    Scalar* kernelRow = kernelInChannel + (j * kX);
                    const Scalar* kernelGradientRow = kernelGradientInChannel + (j * kX);
                    
                    for(int i = 0; i < kX; i++)
                        *(kernelRow + i) -= *(kernelGradientRow + i) * scale;
                }
                biases_[outChan] -= biasGradient_[outChan] * scale;
            }
        }
}



void ConvLayer::maxPool_()
{
    std::size_t channels = activation_.dim(0);
    
    std::size_t outputY = pooled_.dim(1);
    std::size_t outputX = pooled_.dim(2);
    
    std::array<Scalar, stride*stride> candidates;
    
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


void ConvLayer::zeroGradients()
{
    kernelGradient_.fill(0.0f);
    std::fill(biasGradient_.begin(),biasGradient_.end(),0.0f);
}


void ConvLayer::pushCache()
{
    ConvCache cache;
    cache.input = input_;
    cache.activation = activation_;
    cache.maxPoolSource = maxPoolSource_;
    cache_.push(cache);
}


void ConvLayer::popCache()
{
    auto cache = std::move(cache_.front());
    cache_.pop();
    input_ = std::move(cache.input);
    activation_ = std::move(cache.activation);
    maxPoolSource_ = std::move(cache.maxPoolSource);
}


void ConvLayer::initialiseWeights()
{
    const float fanIn = static_cast<float>(inputChannels() * kernelWidth() * kernelHeight());
    
    const float stddev = std::sqrt(2.0f / fanIn);
    
    std::normal_distribution<float> distribution(0.0f, stddev);
    
    for(std::size_t i = 0; i<kernels_.size(); i++)
    {
        kernels_.data()[i] = distribution(rng_);
    }
    
    for(Scalar &bias: biases_) bias = 0.0f;
}
