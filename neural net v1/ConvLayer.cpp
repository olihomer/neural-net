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


std::random_device ConvLayer::rd_;
std::mt19937 ConvLayer::rng_(ConvLayer::rd_());
Scalar ConvLayer::beta1pow;
Scalar ConvLayer::beta2pow;

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
kernelGradient_({outputChannels,inputChannels,3,3}),
kernel_m_({outputChannels,inputChannels,3,3}),
kernel_v_({outputChannels,inputChannels,3,3})
{
    biases_.resize(outputChannels);
    biasGradient_.resize(outputChannels);
    bias_m_.resize(outputChannels);
    bias_v_.resize(outputChannels);

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
                    const std::size_t index = static_cast<std::size_t>(*(maxPoolSourceRow + i));
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
                    indexX += stride;
                }
                indexY += stride;
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
                    const std::size_t index = static_cast<std::size_t>(*(maxPoolSourceRow + i));
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

                    indexX += stride;
                }
                indexY += stride;
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
                        kernel_mRow[i] = beta1 * kernel_mRow[i] + (1 - beta1) * kernelGradientRow[i];
                        kernel_vRow[i] = beta2 * kernel_vRow[i] + (1 - beta2) * kernelGradientRow[i] * kernelGradientRow[i];
                        kernel_mHat = kernel_mRow[i] / (1 - beta1pow);
                        kernel_vHat = kernel_vRow[i] / (1 - beta2pow);
                        kernelRow[i] -= (kernel_mHat / (std::sqrt(kernel_vHat) + epsilon)) * scale;
                        
                       // *(kernelRow + i) -= *(kernelGradientRow + i) * scale;
                        
                    }
                }
            }
            bias_m_[outChan] = beta1 * bias_m_[outChan] + (1 - beta1) * biasGradient_[outChan];
            bias_v_[outChan] = beta2 * bias_v_[outChan] + (1 - beta2) * biasGradient_[outChan] * biasGradient_[outChan];
            bias_mHat = bias_m_[outChan] / (1 - beta1pow);
            bias_vHat = bias_v_[outChan] / (1 - beta2pow);
            
            biases_[outChan] -= (bias_mHat / (std::sqrt(bias_mHat) + epsilon)) * scale;
            
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


void ConvLayer::save(std::ofstream& file) const
{
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
    input_ = Tensor({inChans,InputY,InputX});
    activation_ = Tensor({outChans,InputY,InputX});
    biases_.resize(outChans);
    biasGradient_.resize(outChans);
    pooled_ = Tensor({outChans,InputY/stride,InputX/stride});
    maxPoolSource_ = Tensor({outChans,InputY/stride,InputX/stride});
    cache_ = {};

    //Kernels
    file.read(reinterpret_cast<char*>(kernels_.data()),kernels_.size()*sizeof(Scalar));

    //Biases
    file.read(reinterpret_cast<char*>(biases_.data()),biases_.size()*sizeof(Scalar));

    if(!file)
        throw std::runtime_error("Failed while loading convolution layer");
}
