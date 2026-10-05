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
#include <arm_neon.h>
#include <fstream>
#include "Matrix.hpp"
#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>

std::random_device ConvLayer::rd_;
std::mt19937 ConvLayer::rng_(ConvLayer::rd_());

namespace
{
constexpr bool profileConvLayer = false;
}

ConvLayer::ConvLayer(std::size_t outputChannels,
                     std::size_t inputChannels,
                     std::size_t inputHeight,
                     std::size_t inputWidth,
                     bool bPooling)
:input_({inputChannels,inputHeight,inputWidth}),
inputBatch_({miniBatchSize,inputChannels,inputHeight,inputWidth}),
activation_({outputChannels,inputHeight,inputWidth}),
activationBatch_({miniBatchSize,outputChannels,inputHeight,inputWidth}),
inputGradientBatch_({miniBatchSize,inputChannels,inputHeight,inputWidth}),
pooled_({outputChannels,inputHeight/(bPooling ? stride : 1), inputWidth/(bPooling ? stride : 1)}),
pooledBatch_({miniBatchSize,outputChannels,inputHeight/(bPooling ? stride : 1), inputWidth/(bPooling ? stride : 1)}),
maxPoolSource_({outputChannels, inputHeight/(bPooling ? stride : 1), inputWidth/(bPooling ? stride : 1)}),
maxPoolSourceBatch_({miniBatchSize,outputChannels, inputHeight/(bPooling ? stride : 1), inputWidth/(bPooling ? stride : 1)}),
kernels_({outputChannels,inputChannels,3,3}),
kernelGradient_({outputChannels,inputChannels,3,3}),
kernel_m_({outputChannels,inputChannels,3,3}),
kernel_v_({outputChannels,inputChannels,3,3}),
bPooling_(bPooling),
inputIm2Col_(miniBatchSize * inputHeight * inputWidth, inputChannels * 3 * 3)
{
    biases_.resize(outputChannels);
    biasGradient_.resize(outputChannels);
    bias_m_.resize(outputChannels);
    bias_v_.resize(outputChannels);
    paddedInput_.resize(miniBatchSize * inputChannels * (inputWidth+2) * (inputHeight+2),0.0f);
    paddedInputGradient_.resize(miniBatchSize * inputChannels * (inputWidth+2) * (inputHeight+2),0.0f);

    
    initialiseWeights();
    std::cout << "Constructing ConvLayer with shape " << outputChannels << "," << inputChannels << "," << inputHeight << "," << inputWidth << std::endl;
}

Tensor& ConvLayer::forward (const Tensor& input)
{
    input_ = input;
    convolveIm2Col_();
    
    if(bPooling_)
    {
        maxPool_();
        return pooled_;
    }
    
    return activation_;
}

Tensor& ConvLayer::forwardBatch (const Tensor& input)
{
    inputBatch_ = input;
    convolveIm2ColBatch_();
    
    //At this point, we have convolved the minibatch and the activations are sat in activationsBatch
    
    //We need to separate them back into activations for the rest of the process to work
        
    if(bPooling_)
    {
        maxPoolBatch_();
        return pooledBatch_;
    }
    
    return activationBatch_;
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

Tensor ConvLayer::backward(const Tensor& outputGradient, const bool returnInputGradient, int miniBatchIndex)
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

    const auto inputY = input_.dim(1);
    const auto inputX = input_.dim(2);
    const auto inputChannels = input_.dim(0);
    
    const Scalar* inputData;
    const Scalar* activationData;
    const Scalar* maxPoolSourceData;

    //faster access code

    const Scalar* kernelData = kernels_.data();
    Scalar* kernelGradientData = kernelGradient_.data();
 
    if(miniBatchIndex == -1)
    {
        inputData = input_.data();
        activationData = activation_.data();
        maxPoolSourceData = maxPoolSource_.data();
    }
    else
    {
        inputData = inputBatch_.data() + miniBatchIndex * input_.size();
        activationData = activationBatch_.data() + miniBatchIndex * activation_.size();
        maxPoolSourceData = maxPoolSourceBatch_.data() + miniBatchIndex * maxPoolSource_.size();
    }
    
    const Scalar* outputGradientData = outputGradient.data();

    const std::size_t kernelStride = kernels_.dim(2) * kernels_.dim(3);
    const std::size_t kernelX = kernels_.dim(3);

    //split function into two depending whether we want an input gradient or not

    if(returnInputGradient)
    {
        Tensor inputGradient({input_.dim(0), input_.dim(1), input_.dim(2)});
        Scalar* inputGradientData = inputGradient.data();

        for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
        {
            std::size_t indexY = 0;

            const Scalar* kernelOutChannel = kernelData + (outChan * inputChannels * kernelStride);
            Scalar* kernelGradientOutChannel = kernelGradientData + (outChan * inputChannels * kernelStride);
            const Scalar* activationChannel = activationData + (outChan * inputY * inputX);
            const Scalar* maxPoolSourceChannel = maxPoolSourceData + (outChan * outputY * outputX);
            const Scalar* outputGradientChannel = outputGradientData + (outChan * outputY * outputX);

            for(std::size_t j=0;j<outputY;j++)
            {
                std::size_t indexX = 0;
                const Scalar* maxPoolSourceRow = maxPoolSourceChannel + (j * outputX);
                const Scalar* outputGradientRow = outputGradientChannel + (j * outputX);

                for(std::size_t i=0;i<outputX;i++)
                {
                    //identify winner from activation tensor
                    const std::size_t index = bPooling_ ? static_cast<std::size_t>(*(maxPoolSourceRow + i)) : 0;
                    const std::size_t dx = (index == 2 || index == 0) ? 0 : 1;
                    const std::size_t dy = index < 2 ? 0 : 1;

                    const std::size_t winX = indexX + dx;
                    const std::size_t winY = indexY + dy;
                    const int winXi = static_cast<int>(winX);
                    const int winYi = static_cast<int>(winY);

                    //split out faster loop if not near edges
                    const bool nodanger =
                    winXi > 0 &&
                    winXi < static_cast<int>(inputX) - 1 &&
                    winYi > 0 &&
                    winYi < static_cast<int>(inputY) - 1;

                    if(nodanger)
                    {

                        if(*(activationChannel + (winY * inputX) + winX) > 0.0f)
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
                                Scalar* inputGradientInChannel = inputGradientData + (inChan * inputY * inputX);

                                //unrolled kernel gradient loop

                                Scalar* kg0 = kernelGradientInChannel;
                                Scalar* kg1 = kg0 + 3;
                                Scalar* kg2 = kg1 + 3;

                                const Scalar* in0 = inputInChannel + (winY - 1) * inputX + (winX - 1);
                                const Scalar* in1 = in0 + inputX;
                                const Scalar* in2 = in1 + inputX;

                                kg0[0] += in0[0] * g;
                                kg0[1] += in0[1] * g;
                                kg0[2] += in0[2] * g;

                                kg1[0] += in1[0] * g;
                                kg1[1] += in1[1] * g;
                                kg1[2] += in1[2] * g;

                                kg2[0] += in2[0] * g;
                                kg2[1] += in2[1] * g;
                                kg2[2] += in2[2] * g;

                                //unrolled kernel gradient loop

                                const Scalar* ki0 = kernelInChannel;
                                const Scalar* ki1 = ki0 + 3;
                                const Scalar* ki2 = ki1 + 3;

                                Scalar* ig0 = inputGradientInChannel + (winY - 1) * inputX + (winX - 1);
                                Scalar* ig1 = ig0 + inputX;
                                Scalar* ig2 = ig1 + inputX;

                                ig0[0] += ki0[0] * g;
                                ig0[1] += ki0[1] * g;
                                ig0[2] += ki0[2] * g;

                                ig1[0] += ki1[0] * g;
                                ig1[1] += ki1[1] * g;
                                ig1[2] += ki1[2] * g;

                                ig2[0] += ki2[0] * g;
                                ig2[1] += ki2[1] * g;
                                ig2[2] += ki2[2] * g;
                            }
                            biasGradient_[outChan] += g;
                        }
                    }
                    else
                    {
                        if(*(activationChannel + (winY * inputX) + winX) > 0.0f)
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
                                Scalar* inputGradientInChannel = inputGradientData + (inChan * inputY * inputX);

                                for(int n=-1;n<2;n++)
                                {
                                    Scalar* kernelGradientRow = kernelGradientInChannel + ((n + 1) * kernelX);
                                    const Scalar* kernelRow = kernelInChannel + ((n + 1) * kernelX);

                                    const int inputRowIndex = winYi + n;
                                    if(inputRowIndex < 0 || inputRowIndex >= static_cast<int>(inputY))
                                        continue;

                                    const Scalar* inputRow = inputInChannel + (inputRowIndex * inputX);
                                    Scalar* inputGradientRow = inputGradientInChannel + (inputRowIndex * inputX);

                                    for(int m=-1;m<2;m++)
                                    {
                                        const int inputColIndex = winXi + m;
                                        if(inputColIndex < 0 || inputColIndex >= static_cast<int>(inputX))
                                            continue;

                                        (*(kernelGradientRow + (m + 1))) += (*(inputRow + inputColIndex)) * g;
                                        (*(inputGradientRow + inputColIndex)) += (*(kernelRow + (m + 1))) * g;
                                    }

                                }
                            }
                            biasGradient_[outChan] += g;
                        }
                    }
                    indexX += bPooling_ ? stride : 1;
                }
                indexY += bPooling_ ? stride : 1;
            }
        }
        return inputGradient;
    }
    else //don't need the input gradient
    {
        for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
        {
            std::size_t indexY = 0;

            Scalar* kernelGradientOutChannel = kernelGradientData + (outChan * inputChannels * kernelStride);
            const Scalar* activationChannel = activationData + (outChan * inputY * inputX);
            const Scalar* maxPoolSourceChannel = maxPoolSourceData + (outChan * outputY * outputX);
            const Scalar* outputGradientChannel = outputGradientData + (outChan * outputY * outputX);

            for(std::size_t j=0;j<outputY;j++)
            {
                std::size_t indexX = 0;
                const Scalar* maxPoolSourceRow = maxPoolSourceChannel + (j * outputX);
                const Scalar* outputGradientRow = outputGradientChannel + (j * outputX);

                for(std::size_t i=0;i<outputX;i++)
                {
                    //identify winner from activation tensor
                    const std::size_t index = bPooling_ ? (static_cast<std::size_t>(*(maxPoolSourceRow + i))) : 0;
                    const std::size_t dx = (index == 2 || index == 0) ? 0 : 1;
                    const std::size_t dy = index < 2 ? 0 : 1;

                    const std::size_t winX = indexX + dx;
                    const std::size_t winY = indexY + dy;
                    const int winXi = static_cast<int>(winX);
                    const int winYi = static_cast<int>(winY);

                    //split out faster loop if not near edges
                    const bool nodanger =
                        winXi > 0 &&
                        winXi < static_cast<int>(inputX) - 1 &&
                        winYi > 0 &&
                        winYi < static_cast<int>(inputY) - 1;

                    if(nodanger)
                    {
                        if(*(activationChannel + (winY * inputX) + winX) > 0.0f)
                        {
                            //carry back the gradient. store as a local temp for now to avoid multiple lookups
                            const auto g = *(outputGradientRow + i);

                            // do the unconvolve here!
                            // (winY, winX) is a coordinate of a live preactivation
                            for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                            {
                                Scalar* kernelGradientInChannel = kernelGradientOutChannel + (inChan * kernelStride);
                                const Scalar* inputInChannel = inputData + (inChan * inputY * inputX);

                                //unrolled kernel gradient loop

                                Scalar* kg0 = kernelGradientInChannel;
                                Scalar* kg1 = kg0 + 3;
                                Scalar* kg2 = kg1 + 3;

                                const Scalar* in0 = inputInChannel + (winY - 1) * inputX + winX - 1;
                                const Scalar* in1 = in0 + inputX;
                                const Scalar* in2 = in1 + inputX;

                                kg0[0] += in0[0] * g;
                                kg0[1] += in0[1] * g;
                                kg0[2] += in0[2] * g;

                                kg1[0] += in1[0] * g;
                                kg1[1] += in1[1] * g;
                                kg1[2] += in1[2] * g;

                                kg2[0] += in2[0] * g;
                                kg2[1] += in2[1] * g;
                                kg2[2] += in2[2] * g;

                            }
                            biasGradient_[outChan] += g;
                        }
                    }
                    else
                    {
                        if(*(activationChannel + (winY * inputX) + winX) > 0.0f)
                        {
                            //carry back the gradient. store as a local temp for now to avoid multiple lookups
                            const auto g = *(outputGradientRow + i);

                            // do the unconvolve here!
                            // (winY, winX) is a coordinate of a live preactivation
                            for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                            {
                                Scalar* kernelGradientInChannel = kernelGradientOutChannel + (inChan * kernelStride);
                                const Scalar* inputInChannel = inputData + (inChan * inputY * inputX);

                                for(int n=-1;n<2;n++)
                                {
                                    Scalar* kernelGradientRow = kernelGradientInChannel + ((n + 1) * kernelX);

                                    const int inputRowIndex = winYi + n;
                                    if(inputRowIndex < 0 || inputRowIndex >= static_cast<int>(inputY))
                                        continue;

                                    const Scalar* inputRow = inputInChannel + (inputRowIndex * inputX);

                                    for(int m=-1;m<2;m++)
                                    {
                                        const int inputColIndex = winXi + m;
                                        if(inputColIndex < 0 || inputColIndex >= static_cast<int>(inputX))
                                            continue;

                                        (*(kernelGradientRow + (m + 1))) += (*(inputRow + inputColIndex)) * g;
                                    }

                                }
                            }
                            biasGradient_[outChan] += g;
                        }
                    }

                    indexX += bPooling_ ? stride : 1;
                }
                indexY += bPooling_ ? stride : 1;
            }
        }

        return outputGradient;
    }
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
    
    //same-padding with integer loops to handle edges more easily
    for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
    {
        const Scalar* kernelOutChannel = kernelData + outChan * kernelStride * inputChannels;
        Scalar* activationChannel = activationData + outChan * channelStride;

        //Interior Section using scalar code
        /*
        for(int j = 1; j < inputY-1; j++)
        {
            Scalar* activationRow = activationChannel + (j * inputX);
            for(int i = 1; i < inputX-1; i++)
            {
                Scalar sum = biases_[outChan];
                for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                {
                    const Scalar* channel = inputData + inChan * channelStride;
                    const Scalar* kernelInputChannel = kernelOutChannel + inChan * kernelStride;

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
                *(activationRow + i) = sum > 0.0f ? sum : 0.0f;
            }
        }*/

        //interior section using NEON

        for (int j = 1; j < static_cast<int>(inputY) - 1; ++j)
        {
            Scalar* activationRow = activationChannel + j * inputX;

            int i = 1;

            // Process four neighbouring output pixels at once.
            for (; i + 3 < static_cast<int>(inputX) - 1; i += 4)
            {
                // [bias, bias, bias, bias]
                float32x4_t sums = vdupq_n_f32(biases_[outChan]); //broadcast

                for (std::size_t inChan = 0; inChan < inputChannels; ++inChan)
                {
                    const Scalar* channel =
                        inputData + inChan * channelStride;

                    const Scalar* k =
                        kernelOutChannel + inChan * kernelStride;

                    const Scalar* row0 =
                        channel + (j - 1) * inputX;

                    const Scalar* row1 =
                        channel + j * inputX;

                    const Scalar* row2 =
                        channel + (j + 1) * inputX;

                    // Top kernel row
                    //vfmaq_n_f32(a,b,x) => a = a + b * x

                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row0 + i - 1),
                        k[0]);

                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row0 + i),
                        k[1]);

                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row0 + i + 1),
                        k[2]);

                    // Middle kernel row
                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row1 + i - 1),
                        k[3]);

                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row1 + i),
                        k[4]);

                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row1 + i + 1),
                        k[5]);

                    // Bottom kernel row
                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row2 + i - 1),
                        k[6]);

                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row2 + i),
                        k[7]);

                    sums = vfmaq_n_f32(
                        sums,
                        vld1q_f32(row2 + i + 1),
                        k[8]);
                }

                // ReLU four outputs simultaneously.
                sums = vmaxq_f32(sums, vdupq_n_f32(0.0f));

                // Store four output pixels.
                vst1q_f32(activationRow + i, sums);
            }

            // Scalar tail for the 0–3 interior pixels left over.
            for (; i < static_cast<int>(inputX) - 1; ++i)
            {
                Scalar sum = biases_[outChan];

                for (std::size_t inChan = 0;
                     inChan < inputChannels;
                     ++inChan)
                {
                    const Scalar* channel =
                        inputData + inChan * channelStride;

                    const Scalar* k =
                        kernelOutChannel + inChan * kernelStride;

                    const Scalar* r0 =
                        channel + (j - 1) * inputX + i - 1;

                    const Scalar* r1 =
                        channel + j * inputX + i - 1;

                    const Scalar* r2 =
                        channel + (j + 1) * inputX + i - 1;

                    sum += r0[0] * k[0]
                         + r0[1] * k[1]
                         + r0[2] * k[2]
                         + r1[0] * k[3]
                         + r1[1] * k[4]
                         + r1[2] * k[5]
                         + r2[0] * k[6]
                         + r2[1] * k[7]
                         + r2[2] * k[8];
                }

                activationRow[i] =
                    sum > 0.0f ? sum : 0.0f;
            }
        }

        //edges section

        //top and bottom edges
        for(std::size_t j = 0; j < inputY ; j += (inputY - 1))
        {
            Scalar* activationRow = activationChannel + (j * inputX);
            for(int i = 1; i < inputX - 1; i++)
            {
                Scalar sum = biases_[outChan];
                for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                {
                    const Scalar* channel = inputData + inChan * channelStride;
                    const Scalar* kernelInputChannel = kernelOutChannel + inChan * kernelStride;

                    const Scalar* r0 = (j != 0) ? channel + (j - 1) * inputX + i - 1: channel; // all invalid if j = 0
                    const Scalar* r1 = channel + j * inputX + i - 1; //all valid
                    const Scalar* r2 = (j != inputY - 1) ? channel + (j + 1) * inputX + i - 1: channel; //all invalid if j = inputY - 1

                    if(j==0) // top edge
                    {
                        sum += r1[0] * kernelInputChannel[3]
                            + r1[1] * kernelInputChannel[4]
                            + r1[2] * kernelInputChannel[5]
                            + r2[0] * kernelInputChannel[6]
                            + r2[1] * kernelInputChannel[7]
                            + r2[2] * kernelInputChannel[8];
                    }
                    else //bottom edge
                    {
                        sum += r0[0] * kernelInputChannel[0]
                        + r0[1] * kernelInputChannel[1]
                        + r0[2] * kernelInputChannel[2]
                        + r1[0] * kernelInputChannel[3]
                        + r1[1] * kernelInputChannel[4]
                        + r1[2] * kernelInputChannel[5];
                    }
                }
                *(activationRow + i) = sum > 0.0f ? sum : 0.0f;
            }
        }


        //left and right edges
        for(std::size_t j = 0; j < inputY; j++)
        {
            Scalar* activationRow = activationChannel + (j * inputX);
            for(int i = 0; i < inputX; i+=(inputX - 1))
            {
                Scalar sum = biases_[outChan];
                for(std::size_t inChan=0; inChan < inputChannels; inChan++)
                {
                    const Scalar* channel = inputData + inChan * channelStride;
                    const Scalar* kernelInputChannel = kernelOutChannel + inChan * kernelStride;

                    const std::size_t xBase = (i == 0) ? 0 : static_cast<std::size_t>(i - 1); //increase pointer for left hand side

                    const Scalar* r0 = (j != 0) ? channel + (j - 1) * inputX + xBase: channel; // all invalid if j = 0
                    const Scalar* r1 = channel + j * inputX + xBase; //all valid
                    const Scalar* r2 = (j != inputY - 1) ? channel + (j + 1) * inputX + xBase: channel; //all invalid if j = inputY - 1

                    if(j == 0) // top edge
                    {
                        if(i == 0) // left edge so all the [0] are invalid
                        {
                            sum += r1[0] * kernelInputChannel[4]
                            + r1[1] * kernelInputChannel[5]
                            + r2[0] * kernelInputChannel[7]
                            + r2[1] * kernelInputChannel[8];
                        }
                        else //right edge so all the [2] are invalid
                        {
                            sum += r1[0] * kernelInputChannel[3]
                                + r1[1] * kernelInputChannel[4]
                                + r2[0] * kernelInputChannel[6]
                                + r2[1] * kernelInputChannel[7];
                        }
                    }
                    else if(j == (inputY-1)) //bottom edge
                    {
                        if(i == 0) // left edge so all the [0] are invalid
                        {
                            sum += r0[0] * kernelInputChannel[1]
                            + r0[1] * kernelInputChannel[2]
                            + r1[0] * kernelInputChannel[4]
                            + r1[1] * kernelInputChannel[5];
                        }
                        else //right edge so all the [2] are invalid
                        {
                            sum += r0[0] * kernelInputChannel[0]
                            + r0[1] * kernelInputChannel[1]
                            + r1[0] * kernelInputChannel[3]
                            + r1[1] * kernelInputChannel[4];
                        }
                    }
                    else // middle of the left/right rows
                    {
                        if(i==0) //left so [0] are invalid
                        {
                            sum += r0[0] * kernelInputChannel[1]
                                + r0[1] * kernelInputChannel[2]
                                + r1[0] * kernelInputChannel[4]
                                + r1[1] * kernelInputChannel[5]
                                + r2[0] * kernelInputChannel[7]
                                + r2[1] * kernelInputChannel[8];
                        }
                        else //right so [2] are invalid
                        {
                            sum += r0[0] * kernelInputChannel[0]
                                + r0[1] * kernelInputChannel[1]
                                + r1[0] * kernelInputChannel[3]
                                + r1[1] * kernelInputChannel[4]
                                + r2[0] * kernelInputChannel[6]
                            + r2[1] * kernelInputChannel[7];
                        }

                    }
                }
                *(activationRow + i) = sum > 0.0f ? sum : 0.0f;
            }
        }


    }
}



void ConvLayer::convolvePadded_()
{
    const auto inputChannels = input_.dim(0);
    const auto inputY = input_.dim(1);
    const auto inputX = input_.dim(2);
    const auto outputChannels = kernels_.dim(0);
    
    //faster access code
    
    const Scalar* inputData = input_.data();
    const Scalar* kernelData = kernels_.data();
    Scalar* activationData = activation_.data();
    Scalar* paddedData = paddedInput_.data();
    const std::size_t channelStride = inputY * inputX;
    const std::size_t kernelStride = kernels_.dim(2) * kernels_.dim(3);
    
    //padded buffer
    const std::size_t paddedWidth = inputX + 2;
    const std::size_t paddedChannelStride = (inputY + 2) * paddedWidth;
    
    for(std::size_t inChan = 0; inChan < inputChannels; inChan++)
    {
        const Scalar *inputChannel = inputData + (inChan * channelStride);
        Scalar *paddedInputChannel = paddedData + (inChan * paddedChannelStride);
        
        for(std::size_t i = 0; i < inputY; i++)
        {
            std::memcpy(paddedInputChannel + (i+1) * (inputX+2) + 1, inputChannel + i * inputX, inputX * sizeof(Scalar));
        }
    }
    
   
    
    //same-padding with integer loops to handle edges more easily
    for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
    {
        const Scalar* kernelOutChannel = kernelData + outChan * kernelStride * inputChannels;
        Scalar* activationChannel = activationData + outChan * channelStride;
        
        
        //interior section using NEON
        
        for (int j = 0; j < static_cast<int>(inputY); ++j)
        {
            Scalar* activationRow = activationChannel + j * inputX;
            
            int i = 0;
            
            // Process four neighbouring output pixels at once.
            for (; i + 3 < static_cast<int>(inputX); i += 4)
            {
                // [bias, bias, bias, bias]
                float32x4_t sums = vdupq_n_f32(biases_[outChan]); //broadcast
                
                for (std::size_t inChan = 0; inChan < inputChannels; ++inChan)
                {
                    const Scalar* channel =
                    paddedData + inChan * paddedChannelStride;
                    
                    const Scalar* k =
                    kernelOutChannel + inChan * kernelStride;
                    
                    const Scalar* row0 =
                    channel + j * paddedWidth;
                    
                    const Scalar* row1 =
                    row0 + paddedWidth;
                    
                    const Scalar* row2 =
                    row1 + paddedWidth;
                    
                    // Top kernel row
                    //vfmaq_n_f32(a,b,x) => a = a + b * x
                    
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row0 + i),
                                       k[0]);
                    
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row0 + i + 1),
                                       k[1]);
                    
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row0 + i + 2),
                                       k[2]);
                    
                    // Middle kernel row
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row1 + i),
                                       k[3]);
                    
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row1 + i + 1),
                                       k[4]);
                    
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row1 + i + 2),
                                       k[5]);
                    
                    // Bottom kernel row
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row2 + i),
                                       k[6]);
                    
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row2 + i + 1),
                                       k[7]);
                    
                    sums = vfmaq_n_f32(
                                       sums,
                                       vld1q_f32(row2 + i + 2),
                                       k[8]);
                }
                
                // ReLU four outputs simultaneously.
                sums = vmaxq_f32(sums, vdupq_n_f32(0.0f));
                
                // Store four output pixels.
                vst1q_f32(activationRow + i, sums);
            }
            
            // Scalar tail for the 0–3 interior pixels left over.
            for (; i < static_cast<int>(inputX); ++i)
            {
                Scalar sum = biases_[outChan];
                
                for (std::size_t inChan = 0;
                     inChan < inputChannels;
                     ++inChan)
                {
                    const Scalar* channel =
                    paddedData + inChan * paddedChannelStride;
                    
                    const Scalar* k =
                    kernelOutChannel + inChan * kernelStride;
                    
                    const Scalar* r0 =
                    channel + j + paddedWidth + i;
                    
                    const Scalar* r1 =
                    r0 + paddedWidth;
                    
                    const Scalar* r2 =
                    r1 + paddedWidth;
                    
                    sum += r0[0] * k[0]
                    + r0[1] * k[1]
                    + r0[2] * k[2]
                    + r1[0] * k[3]
                    + r1[1] * k[4]
                    + r1[2] * k[5]
                    + r2[0] * k[6]
                    + r2[1] * k[7]
                    + r2[2] * k[8];
                }
                
                activationRow[i] =
                sum > 0.0f ? sum : 0.0f;
            }
        }
        
        
    }

}



void ConvLayer::gradient_descent(std::size_t batchSize, const Scalar learningRate)
{
    const auto outputChannels = kernels_.dim(0);
    const auto inputChannels = kernels_.dim(1);
    const auto kY = kernels_.dim(2);
    const auto kX = kernels_.dim(3);
    const auto kernelStride = kY * kX;
    
    //faster access code
    Scalar* kernelData = kernels_.data();
    const Scalar* kernelGradientData = kernelGradient_.data();
    
    //Adam parameters
    Scalar* kernel_mData = kernel_m_.data();
    Scalar* kernel_vData = kernel_v_.data();
    Scalar kernel_mHat;
    Scalar kernel_vHat;
    Scalar bias_mHat;
    Scalar bias_vHat;
    
    //Adam algorithm
    // m = beta1 * prev m + (1 - beta1) * gradient
    // v = beta2 * prev v + (1 - beta2) * gradient * gradient
    // mhat = m / (1 - beta1^t)
    // vhat = v / (1 - beta2^t)
    // w = prev w - mhat / (sqrt (vhat) + epsilon) * alpha

        for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
        {
            Scalar* kernelOutChannel = kernelData + (outChan * inputChannels * kernelStride);
            Scalar* kernel_mOutChannel = kernel_mData + (outChan * inputChannels * kernelStride);
            Scalar* kernel_vOutChannel = kernel_vData + (outChan * inputChannels * kernelStride);
            
            const Scalar* kernelGradientOutChannel = kernelGradientData + (outChan * inputChannels * kernelStride);

            for(std::size_t inChan = 0; inChan < inputChannels; inChan++)
            {
                Scalar* kernelInChannel = kernelOutChannel + (inChan * kernelStride);
                Scalar* kernel_mInChannel = kernel_mOutChannel + (inChan * kernelStride);
                Scalar* kernel_vInChannel = kernel_vOutChannel + (inChan * kernelStride);
                
                const Scalar* kernelGradientInChannel = kernelGradientOutChannel + (inChan * kernelStride);

                for(int j = 0; j < kY; j++)
                {
                    Scalar* kernelRow = kernelInChannel + (j * kX);
                    Scalar* kernel_mRow = kernel_mInChannel + (j * kX);
                    Scalar* kernel_vRow = kernel_vInChannel + (j * kX);

                    const Scalar* kernelGradientRow = kernelGradientInChannel + (j * kX);

                    for(int i = 0; i < kX; i++)
                    {
                        const Scalar g = kernelGradientRow[i];
                        kernel_mRow[i] = beta1 * kernel_mRow[i] + (1 - beta1) * g;
                        kernel_vRow[i] = beta2 * kernel_vRow[i] + (1 - beta2) * g * g;
                        kernel_mHat = kernel_mRow[i] / (1 - beta1pow);
                        kernel_vHat = kernel_vRow[i] / (1 - beta2pow);
                        kernelRow[i] -= (kernel_mHat / (std::sqrt(kernel_vHat) + epsilon)) * learningRate;
                        
                       // *(kernelRow + i) -= *(kernelGradientRow + i) * scale;
                        
                    }
                }
            }
            const Scalar g = biasGradient_[outChan];
            bias_m_[outChan] = beta1 * bias_m_[outChan] + (1 - beta1) * g;
            bias_v_[outChan] = beta2 * bias_v_[outChan] + (1 - beta2) * g * g;
            bias_mHat = bias_m_[outChan] / (1 - beta1pow);
            bias_vHat = bias_v_[outChan] / (1 - beta2pow);
            
            biases_[outChan] -= (bias_mHat / (std::sqrt(bias_vHat) + epsilon)) * learningRate;
            
            //biases_[outChan] -= biasGradient_[outChan] * scale;
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


void ConvLayer::maxPoolBatch_()
{
    std::size_t channels = activation_.dim(0);

    std::size_t outputY = pooled_.dim(1);
    std::size_t outputX = pooled_.dim(2);

    std::array<Scalar, stride*stride> candidates;

    for(std::size_t batchIndex = 0; batchIndex < miniBatchSize; batchIndex++)
    {
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
                            candidates[x+y*stride] = activationBatch_(batchIndex, chan,indexY + y,indexX + x);
                        }
                    auto max = std::max_element(candidates.begin(), candidates.end());
                    pooledBatch_(batchIndex,chan,j,i) = *max;
                    maxPoolSourceBatch_(batchIndex,chan,j,i) = max - candidates.begin();
                    indexX += stride;
                }
                indexY += stride;
                indexX = 0;
            }
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

void ConvLayer::pushCacheBatch(const std::size_t batchSize)
{
    ConvCache cache;
    cache.activation = activationBatch_;
    cache.maxPoolSource = maxPoolSourceBatch_;
    cache.batchSize = batchSize;
    cache.inputIm2Col = inputIm2Col_;
    cache_.push(std::move(cache));
    cacheSize++;
}


void ConvLayer::popCache()
{
    auto cache = std::move(cache_.front());
    cache_.pop();
    input_ = std::move(cache.input);
    activation_ = std::move(cache.activation);
    maxPoolSource_ = std::move(cache.maxPoolSource);
}

std::size_t ConvLayer::popCacheBatch()
{
    auto cache = std::move(cache_.front());
    cache_.pop();
    activationBatch_ = std::move(cache.activation);
    maxPoolSourceBatch_ = std::move(cache.maxPoolSource);
    inputIm2Col_ = std::move(cache.inputIm2Col);
    cacheSize--;
    return cache.batchSize;
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


void ConvLayer::save(std::ofstream& file) const
{
    //Pooling flag
    file.write(reinterpret_cast<const char*>(&bPooling_),sizeof(bPooling_));
    
    //Output channels
    auto outChans = getOutputChannels();
    file.write(reinterpret_cast<const char*>(&outChans),sizeof(outChans));

    //Input channels
    auto inChans = inputChannels();
    file.write(reinterpret_cast<const char*>(&inChans),sizeof(inChans));

    //Input height
    auto inputY = inputHeight();
    file.write(reinterpret_cast<const char*>(&inputY),sizeof(inputY));

    //Input width
    auto inputX = inputWidth();
    file.write(reinterpret_cast<const char*>(&inputX),sizeof(inputX));

    //Kernels
    file.write(reinterpret_cast<const char*>(kernels_.data()),kernels_.size()*sizeof(Scalar));

    //Biases
    file.write(reinterpret_cast<const char*>(biases_.data()),biases_.size()*sizeof(Scalar));
}

void ConvLayer::load(std::ifstream& file)
{
    //Pooling flag
    file.read(reinterpret_cast<char*>(&bPooling_),sizeof(bPooling_));

    //Output channels
    std::size_t outChans;
    file.read(reinterpret_cast<char*>(&outChans),sizeof(outChans));

    //Input channels
    std::size_t inChans;
    file.read(reinterpret_cast<char*>(&inChans),sizeof(inChans));

    //Input height
    std::size_t InputY;
    file.read(reinterpret_cast<char*>(&InputY),sizeof(InputY));

    //Input width
    std::size_t InputX;
    file.read(reinterpret_cast<char*>(&InputX),sizeof(InputX));

    kernels_ = Tensor({outChans,inChans,3,3});
    kernelGradient_ = Tensor({outChans,inChans,3,3});
    kernel_m_ = Tensor({outChans,inChans,3,3});
    kernel_v_ = Tensor({outChans,inChans,3,3});
    
    input_ = Tensor({inChans,InputY,InputX});
    activation_ = Tensor({outChans,InputY,InputX});
    biases_.resize(outChans);
    biasGradient_.resize(outChans);
    bias_m_.resize(outChans);
    bias_v_.resize(outChans);
    paddedInput_.resize((InputX+2)*(InputY+2));
    pooled_ = Tensor({outChans,InputY/(bPooling_?stride:1),InputX/(bPooling_?stride:1)});
    maxPoolSource_ = Tensor({outChans,InputY/(bPooling_?stride:1),InputX/(bPooling_?stride:1)});
    cache_ = {};

    //Kernels
    file.read(reinterpret_cast<char*>(kernels_.data()),kernels_.size()*sizeof(Scalar));

    //Biases
    file.read(reinterpret_cast<char*>(biases_.data()),biases_.size()*sizeof(Scalar));

    if(!file)
        throw std::runtime_error("Failed while loading convolution layer");
}



void ConvLayer::convolveIm2ColBatch_()
{
    const auto inputChannels = inputBatch_.dim(1);
    const auto inputY = inputBatch_.dim(2);
    const auto inputX = inputBatch_.dim(3);
    const auto outputChannels = kernels_.dim(0);
    
    Tensor gemmOutput({miniBatchSize,outputChannels,inputX,inputY});
    
    //faster access code
    
    const Scalar* inputData = inputBatch_.data();
    
    const std::size_t channelStride = inputY * inputX;
    const std::size_t kernelStride = kernels_.dim(2) * kernels_.dim(3);
    const std::size_t inputMatrixCols = inputChannels * kernelStride;
    
    
    //padded buffer
    Scalar* paddedData = paddedInput_.data();
    const std::size_t paddedWidth = inputX + 2;
    const std::size_t paddedChannelStride = (inputY + 2) * paddedWidth;
    const std::size_t inputBatchStride = channelStride * inputChannels;
    const std::size_t paddedBatchStride = paddedChannelStride * inputChannels;
    const std::size_t im2colRows = miniBatchSize * channelStride;
    
    for(std::size_t batchIndex = 0; batchIndex < miniBatchSize; batchIndex++)
    {
        const Scalar* inputBatch = inputData + batchIndex * inputBatchStride;
        Scalar* paddedInputBatch = paddedData + batchIndex * paddedBatchStride;
        
        for(std::size_t inChan = 0; inChan < inputChannels; inChan++)
        {
            const Scalar* inputChannel = inputBatch + (inChan * channelStride);
            Scalar* paddedInputChannel = paddedInputBatch + (inChan * paddedChannelStride);
            
            for(std::size_t i = 0; i < inputY; i++)
            {
                std::memcpy(paddedInputChannel + (i+1) * (inputX+2) + 1, inputChannel + i * inputX, inputX * sizeof(Scalar));
            }
        }
    }
    
    //Im2Col
    // Need to create two matrices
    // First matrix is the inputs divided into patches same size as kernel (3x3)
    // One row for each position on the input image, i.e. rows = InputY x Input X
    // Each row contains InputChannels x 3 x 3 elements, i.e. columns = InputChannels x kernelStride
    
    //This now sits as a member of the class: Matrix inputIm2Col_(inputX * inputY, inputChannels * kernelStride);
    
    //To turn this into a batch operation, we need to change the number of rows to miniBatchSize * inputX * inputY
    
    // populate input matrix from paddedData
    
    std::size_t destIndex = 0; //points to current cell of matrix to be populated
    Scalar* inputMatrixData = inputIm2Col_.data(); //points to inputMatrix
    
    for(std::size_t batchIndex = 0; batchIndex < miniBatchSize; batchIndex++) //walk through minibatch
    {
        Scalar* currentBatchPaddedData = paddedData + batchIndex * paddedBatchStride;
        Scalar* currentChanPaddedData = currentBatchPaddedData;
        
        for(std::size_t posIndexY = 0; posIndexY < inputY; posIndexY++) //walks down every actual image column
        {
            for(std::size_t posIndexX = 0; posIndexX < inputX; posIndexX++) // walks across every actual image row
            {
                const std::size_t sourceIndex = posIndexY * paddedWidth + posIndexX;
                for(std::size_t inChan = 0; inChan < inputChannels; inChan++) //walks through all inputchannels
                {
                    // adds nine points at a time to the matrix
                    const Scalar* row0 = currentChanPaddedData + sourceIndex;
                    const Scalar* row1 = row0 + paddedWidth;
                    const Scalar* row2 = row1 + paddedWidth;
                    
                    inputMatrixData[destIndex++] = row0[0];
                    inputMatrixData[destIndex++] = row0[1];
                    inputMatrixData[destIndex++] = row0[2];
                    inputMatrixData[destIndex++] = row1[0];
                    inputMatrixData[destIndex++] = row1[1];
                    inputMatrixData[destIndex++] = row1[2];
                    inputMatrixData[destIndex++] = row2[0];
                    inputMatrixData[destIndex++] = row2[1];
                    inputMatrixData[destIndex++] = row2[2];
                    
                    currentChanPaddedData += paddedChannelStride; // on to next channel
                }
                currentChanPaddedData = currentBatchPaddedData;
            }
            currentChanPaddedData = currentBatchPaddedData;
        }
    }
    //second matrix is the our existing kernel tensor with dimensions
    //now just need to do kernels_ * inputIm2Col T -> activation
    
    //std::cout << "About to sgemm. Kernels Matrix A: " << std::endl;
    //kernels_.print();
    //std::cout << "Inputs Matrix B. Rows = " << inputIm2Col_.rows() << " Cols = " << inputIm2Col_.cols() << " Size = " << inputIm2Col_.size() << std::endl;
    
    
    cblas_sgemm(
                CblasRowMajor, //order of matrices
                CblasNoTrans, //transpose matrix A?
                CblasTrans, //transpose matrix B?
                outputChannels, //rows in matrix A
                im2colRows, //columns in matrix B => inputX * InputY * miniBatchSize
                inputMatrixCols, //columns in matrix A & rows in matrix B => inputChannels * 9
                1.0f, //scaling factor
                kernels_.data(), //matrix A
                inputMatrixCols, //first dimension of matrix A => inputChannels * 9
                inputIm2Col_.data(), // matrix B
                inputMatrixCols, //first dimension of matrix B => inputChannels * 9
                0.0f, //scaling factor for Matrix C
                gemmOutput.data(), // matrix C
                im2colRows // first dimension of matrix C => inputX * InputY * miniBatchSize
                );
    
    //rearrange gemm data from [O][N][H][W] -> [N][O][H][W]
    
    const std::size_t P = inputX * inputY;
    const std::size_t R = miniBatchSize * P;
    
    for(std::size_t n = 0; n < miniBatchSize; n++)
    {
        for(std::size_t o = 0; o < outputChannels; o++)
        {
            const Scalar* src = gemmOutput.data() + o * R + n * P;
            
            Scalar* dst = activationBatch_.data() + (n * outputChannels + o) * P;
            
            for(std::size_t p = 0; p < P; p++)
                dst[p] = std::max(0.0f, src[p] + biases_[o]);
        }
    }
    
    //std::cout << "End of Convolve()" << std::endl;
}
            

void ConvLayer::convolveIm2Col_()
{
    const auto inputChannels = input_.dim(0);
    const auto inputY = input_.dim(1);
    const auto inputX = input_.dim(2);
    const auto outputChannels = kernels_.dim(0);
    
    //faster access code
    
    const Scalar* inputData = input_.data();
    const Scalar* kernelData = kernels_.data();

    const std::size_t channelStride = inputY * inputX;
    const std::size_t kernelStride = kernels_.dim(2) * kernels_.dim(3);
    const std::size_t inputMatrixCols = inputChannels * kernelStride;
    
    
    //padded buffer
    Scalar* paddedData = paddedInput_.data();
    const std::size_t paddedWidth = inputX + 2;
    const std::size_t paddedChannelStride = (inputY + 2) * paddedWidth;
    
    for(std::size_t inChan = 0; inChan < inputChannels; inChan++)
    {
        const Scalar *inputChannel = inputData + (inChan * channelStride);
        Scalar *paddedInputChannel = paddedData + (inChan * paddedChannelStride);
        
        for(std::size_t i = 0; i < inputY; i++)
        {
            std::memcpy(paddedInputChannel + (i+1) * (inputX+2) + 1, inputChannel + i * inputX, inputX * sizeof(Scalar));
        }
    }
    
    
    //Im2Col
    // Need to create two matrices
    // First matrix is the inputs divided into patches same size as kernel (3x3)
    // One row for each position on the input image, i.e. rows = InputY x Input X
    // Each row contains InputChannels x 3 x 3 elements, i.e. columns = InputChannels x kernelStride
    
    //Matrix inputIm2Col(inputX * inputY, inputMatrixCols);
    
    // populate input matrix from paddedData
    
    std::size_t sourceIndex = 0; //points to current top-left position on paddedData
    std::size_t destIndex = 0; //points to current cell of matrix to be populated
    Scalar* inputMatrixData = inputIm2Col_.data(); //points to inputMatrix
    
    Scalar* currentChanPaddedData = paddedData;
    
    for(std::size_t posIndexY = 0; posIndexY < inputY; posIndexY++) //walks down every actual image column
    {
        for(std::size_t posIndexX = 0; posIndexX < inputX; posIndexX++) // walks across every actual image row
        {
            for(std::size_t inChan = 0; inChan < inputChannels; inChan++) //walks through all inputchannels
            {
                // adds nine points at a time to the matrix
                const Scalar* row0 = currentChanPaddedData + sourceIndex;
                const Scalar* row1 = row0 + paddedWidth;
                const Scalar* row2 = row1 + paddedWidth;
                
                inputMatrixData[destIndex++] = row0[0];
                inputMatrixData[destIndex++] = row0[1];
                inputMatrixData[destIndex++] = row0[2];
                inputMatrixData[destIndex++] = row1[0];
                inputMatrixData[destIndex++] = row1[1];
                inputMatrixData[destIndex++] = row1[2];
                inputMatrixData[destIndex++] = row2[0];
                inputMatrixData[destIndex++] = row2[1];
                inputMatrixData[destIndex++] = row2[2];
                
                currentChanPaddedData += paddedChannelStride; // on to next channel
            }
            sourceIndex++; //step along row by one pixel
            currentChanPaddedData = paddedData;
        }
        sourceIndex += 2;
        currentChanPaddedData = paddedData;
    }
    
    //second matrix is the our existing kernel tensor
    //now just need to do kernels_ * inputIm2Col T -> activation
    
    cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                outputChannels,
                channelStride,
                inputMatrixCols,
                1.0f,
                kernels_.data(),
                inputMatrixCols,
                inputIm2Col_.data(),
                inputMatrixCols,
                0.0f,
                activation_.data(),
                channelStride
                );
    
    
    for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
        for(std::size_t i = 0; i < inputX*inputY; i++)
        {
            Scalar& value = activation_.data()[outChan * channelStride + i];
            value = std::max(0.0f, value + biases_[outChan]);
        }
    
}



void ConvLayer::unPool(const Scalar *activationData, Scalar *&activationGradientData, Matrix &activationGradients, std::size_t inputX, std::size_t inputY, const Scalar *maxPoolSourceData, std::size_t outputChannels, const GradientView &outputGradient, std::size_t outputX, std::size_t outputY, int thisMiniBatchSize) {
    activationGradients = Matrix(outputChannels, thisMiniBatchSize * inputY * inputX);
    activationGradientData = activationGradients.data();
    
    // unpool/unrelu and put the gradient into activationGradientBatch
    
    for(std::size_t n = 0; n < thisMiniBatchSize; n++)
    {
        const Scalar* maxPoolSourceBatch = maxPoolSourceData + (n * outputChannels  * outputY * outputX);
        const Scalar* activationBatch = activationData + (n * outputChannels * inputY * inputX);
        
        
        for(std::size_t outChan = 0; outChan < outputChannels; outChan++)
        {
            std::size_t indexY = 0;
            const Scalar* maxPoolSourceChannel = maxPoolSourceBatch + (outChan * outputY * outputX);
            Scalar* activationGradientChannel = activationGradientData + outChan * (thisMiniBatchSize * inputX * inputY) + n * (inputX * inputY);
            const Scalar* activationChannel = activationBatch + (outChan * inputY * inputX);
            
            for(std::size_t j = 0; j < outputY; j++)
            {
                std::size_t indexX = 0;
                const Scalar* maxPoolSourceRow = maxPoolSourceChannel + (j * outputX);
                
                for(std::size_t i = 0; i < outputX; i++)
                {
                    //identify winner from activation tensor
                    const std::size_t index = bPooling_ ? static_cast<std::size_t>(*(maxPoolSourceRow + i)) : 0;
                    const std::size_t dx = (index == 2 || index == 0) ? 0 : 1;
                    const std::size_t dy = index < 2 ? 0 : 1;
                    
                    const std::size_t winX = indexX + dx;
                    const std::size_t winY = indexY + dy;
                    
                    if(*(activationChannel + (winY * inputX) + winX) > 0.0f)
                        activationGradientChannel[winY * inputX + winX] = outputGradient(n,outChan*outputX*outputY + j * outputX + i);
                    
                    indexX += bPooling_ ? stride : 1;
                }
                indexY += bPooling_ ? stride : 1;
            }
        }
    }
}

void ConvLayer::biasGradients(Scalar *activationGradientData, std::size_t gradientWidth, std::size_t outputChannels) {
    for(std::size_t o = 0; o < outputChannels; o++)
    {
        const Scalar* row = activationGradientData + o * gradientWidth;
        
        Scalar sum = 0.0f;
        
        for(std::size_t i = 0; i< gradientWidth; i++)
            sum += row[i];
        
        biasGradient_[o] += sum;
    }
}

void ConvLayer::Col2Im(std::size_t inputChannels, Matrix &inputGradient, std::size_t inputX, std::size_t inputY, std::size_t &paddedBatchStride, std::size_t &paddedChannelStride, std::size_t &paddedWidth, int thisMiniBatchSize) {
    std::size_t sourceIndex = 0; //points to current top-left position on paddedData
    std::size_t destIndex = 0; //points to current cell of matrix to be populated
    Scalar* inputMatrixData = inputGradient.data(); //points to inputGradient matrix
    
    Scalar* paddedData = paddedInputGradient_.data();
    paddedWidth = inputX + 2;
    paddedChannelStride = (inputY + 2) * (inputX + 2);
    paddedBatchStride = paddedChannelStride  * inputChannels;
    
    std::fill_n(paddedInputGradient_.data(),thisMiniBatchSize * paddedBatchStride,0.0f);
    
    for(std::size_t batchIndex = 0; batchIndex < thisMiniBatchSize; batchIndex++) //walk through minibatch
    {
        Scalar* batch = paddedData + batchIndex * paddedBatchStride;
        
        for(std::size_t posIndexY = 0; posIndexY < inputY; posIndexY++) //walks down every actual image column
        {
            for(std::size_t posIndexX = 0; posIndexX < inputX; posIndexX++) // walks across every actual image row
            {
                const std::size_t sourceIndex = posIndexY * paddedWidth + posIndexX;
                for(std::size_t inChan = 0; inChan < inputChannels; inChan++) //walks through all inputchannels
                {
                    // adds nine points at a time to the matrix
                    Scalar* patch = batch + inChan * paddedChannelStride + sourceIndex;
                    
                    Scalar* row0 = patch;
                    Scalar* row1 = row0 + paddedWidth;
                    Scalar* row2 = row1 + paddedWidth;
                    
                    row0[0] += inputMatrixData[destIndex++] ;
                    row0[1] += inputMatrixData[destIndex++];
                    row0[2] += inputMatrixData[destIndex++];
                    row1[0] += inputMatrixData[destIndex++];
                    row1[1] += inputMatrixData[destIndex++];
                    row1[2] += inputMatrixData[destIndex++];
                    row2[0] += inputMatrixData[destIndex++];
                    row2[1] += inputMatrixData[destIndex++];
                    row2[2] += inputMatrixData[destIndex++];
                    
                }
            }
        }
    }
}

void ConvLayer::unPad(std::size_t inputChannels, std::size_t inputX, std::size_t inputY, std::size_t paddedBatchStride, std::size_t paddedChannelStride, std::size_t paddedWidth, int thisMiniBatchSize) {
    for(std::size_t batchIndex = 0; batchIndex < thisMiniBatchSize; batchIndex++) //walk through minibatch
    {
        for(std::size_t c = 0; c < inputChannels ; c++)
        {
            Scalar *dst = inputGradientBatch_.data() + batchIndex * inputChannels * inputX * inputY + c * inputX * inputY;
            Scalar *src = paddedInputGradient_.data() + batchIndex * paddedBatchStride + c * paddedChannelStride;
            
            for(std::size_t j = 0; j < inputY; j++)
            {
                memcpy(dst + j * inputX, src + (j+1) * paddedWidth + 1, inputX * sizeof(Scalar));
                src += (inputX + 2);
                dst += inputX;
            }
        }
    }
}

GradientView ConvLayer::backwardBatch(const GradientView& outputGradient, const bool returnInputGradient, int thisMiniBatchSize)
{
    //unpool
    //Relu derivative
    //unconvolve
    // - kernel gradients
    // - bias gradients
    // - input gradients
    // return input gradients
    
    const std::size_t outputChannels = getOutputChannels();
    const std::size_t outputY = getOutputHeight();
    const std::size_t outputX = getOutputWidth();
    
    const auto inputY = input_.dim(1);
    const auto inputX = input_.dim(2);
    const auto inputChannels = input_.dim(0);
    
    const std::size_t kernelY = kernels_.dim(2);
    const std::size_t kernelX = kernels_.dim(3);
    
    //these all have a mini batch worth of data popped into them
    const Scalar* activationData = activationBatch_.data();
    const Scalar* maxPoolSourceData = maxPoolSourceBatch_.data();
    
    Matrix activationGradients;
    Scalar * activationGradientData;
    
    unPool(activationData, activationGradientData, activationGradients, inputX, inputY, maxPoolSourceData, outputChannels, outputGradient, outputX, outputY, thisMiniBatchSize);
 
    const std::size_t gradientWidth = thisMiniBatchSize * inputY * inputX;
    
    biasGradients(activationGradientData, gradientWidth, outputChannels);
    
    //activationGradients now has the right gradients in the right places and is a matrix with [O, NxP] = dY
    // inputIm2Col is [NxP,K] = Xcol
    // now need to calculate dW = dY x Xcol
    // [O, NxP] x [NxP, K] = [O,K]
    //  dY = A    Xcol = B
    
    cblas_sgemm(
                CblasRowMajor, //order of matrices
                CblasNoTrans, //transpose matrix A?
                CblasNoTrans, //transpose matrix B?
                outputChannels, //rows in matrix A
                inputChannels * kernelX * kernelY, //columns in matrix B
                thisMiniBatchSize * inputX * inputY, //columns in matrix A & rows in matrix B
                1.0f, //scaling factor
                activationGradients.data(), //matrix A
                thisMiniBatchSize * inputX * inputY, //first dimension of matrix A
                inputIm2Col_.data(), // matrix B
                inputChannels * kernelX * kernelY, //first dimension of matrix B
                1.0f, //scaling factor for Matrix C
                kernelGradient_.data(), // matrix C
                inputChannels * kernelX * kernelY // first dimension of matrix C => inputX * InputY * miniBatchSize
                );
    
    if(returnInputGradient)
    {
        
        // Now need to calculate the input gradient
        // dXcol = dY T x W
        // [NxP, K] = [NxP, O] x [O, K]
        //     C          A T        B
        
        Matrix inputGradient(thisMiniBatchSize * inputX * inputY, inputChannels * kernelX * kernelY);
        
        cblas_sgemm(
                    CblasRowMajor, //order of matrices
                    CblasTrans, //transpose matrix A?
                    CblasNoTrans, //transpose matrix B?
                    thisMiniBatchSize * inputX * inputY, //rows in matrix A
                    inputChannels * kernelX * kernelY, //columns in matrix B
                    outputChannels, //columns in matrix A & rows in matrix B
                    1.0f, //scaling factor
                    activationGradients.data(), //matrix A
                    thisMiniBatchSize * inputX * inputY, //first dimension of matrix A
                    kernels_.data(), // matrix B
                    inputChannels * kernelX * kernelY, //first dimension of matrix B
                    0.0f, //scaling factor for Matrix C
                    inputGradient.data(), // matrix C
                    inputChannels * kernelX * kernelY // first dimension of matrix C => inputX * InputY * miniBatchSize
                    );
        
        
        
        //Col2IM
        // inputGradient now contains the input gradients in a [NxP, K] matrix, just like we got from Im2Col
        // i.e. inputs divided into patches same size as kernel (3x3)
        // One row for each position on the input image and batch example, i.e. rows = InputY x Input X x miniBatchSize
        // Each row contains InputChannels x 3 x 3 elements, i.e. columns = InputChannels x kernelStride
        
        std::size_t paddedWidth;
        std::size_t paddedChannelStride;
        std::size_t paddedBatchStride;
        
        Col2Im(inputChannels, inputGradient, inputX, inputY, paddedBatchStride, paddedChannelStride, paddedWidth, thisMiniBatchSize);
        
        //paddedInputGradient should now contain the inputGradient data
        //just need to unpad it and store in inputGradientBatch
        
        unPad(inputChannels, inputX, inputY, paddedBatchStride, paddedChannelStride, paddedWidth, thisMiniBatchSize);
        
        return GradientView(inputGradientBatch_.data(), thisMiniBatchSize * input_.size(), input_.size(), 1);
    }
    return outputGradient;
}
