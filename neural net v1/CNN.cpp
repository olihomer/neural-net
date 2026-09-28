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
#include <algorithm>
#include <fstream>
#include <cmath>
#include <array>


namespace
{
    constexpr bool profileCNN = false;
    constexpr std::size_t rotationImageSide = 28;
    constexpr std::size_t rotationImageSize = rotationImageSide * rotationImageSide;
    constexpr Scalar pi = 3.14159265358979323846f;
    constexpr std::array<int, 7> rotationDegrees = {-5, 0, 5};

    struct RotationContribution
    {
        std::size_t targetIndex = 0;
        Scalar weight = 0.0f;
    };

    struct SourcePixelRotation
    {
        std::array<RotationContribution, 4> contributions;
        std::size_t count = 0;
    };

    struct RotationLookup
    {
        int degrees = 0;
        std::array<SourcePixelRotation, rotationImageSize> sourcePixels;
    };

    void addContribution(SourcePixelRotation& sourcePixel, int targetX, int targetY, Scalar weight)
    {
        if(weight <= 0.0f ||
           targetX < 0 ||
           targetY < 0 ||
           targetX >= static_cast<int>(rotationImageSide) ||
           targetY >= static_cast<int>(rotationImageSide))
        {
            return;
        }

        sourcePixel.contributions[sourcePixel.count] = {
            static_cast<std::size_t>(targetY) * rotationImageSide + static_cast<std::size_t>(targetX),
            weight
        };
        sourcePixel.count++;
    }

    RotationLookup makeRotationLookup(int degrees)
    {
        RotationLookup lookup;
        lookup.degrees = degrees;

        const Scalar theta = static_cast<Scalar>(degrees) * pi / 180.0f;
        const Scalar cosTheta = std::cos(theta);
        const Scalar sinTheta = std::sin(theta);
        const Scalar centre = static_cast<Scalar>(rotationImageSide - 1) / 2.0f;

        for(std::size_t sourceY = 0; sourceY < rotationImageSide; sourceY++)
        {
            for(std::size_t sourceX = 0; sourceX < rotationImageSide; sourceX++)
            {
                const Scalar x = static_cast<Scalar>(sourceX) - centre;
                const Scalar y = static_cast<Scalar>(sourceY) - centre;
                const Scalar targetX = cosTheta * x - sinTheta * y + centre;
                const Scalar targetY = sinTheta * x + cosTheta * y + centre;

                const int x0 = static_cast<int>(std::floor(targetX));
                const int y0 = static_cast<int>(std::floor(targetY));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const Scalar xWeight = targetX - static_cast<Scalar>(x0);
                const Scalar yWeight = targetY - static_cast<Scalar>(y0);

                SourcePixelRotation& sourcePixel = lookup.sourcePixels[sourceY * rotationImageSide + sourceX];
                addContribution(sourcePixel, x0, y0, (1.0f - xWeight) * (1.0f - yWeight));
                addContribution(sourcePixel, x1, y0, xWeight * (1.0f - yWeight));
                addContribution(sourcePixel, x0, y1, (1.0f - xWeight) * yWeight);
                addContribution(sourcePixel, x1, y1, xWeight * yWeight);
            }
        }

        return lookup;
    }

    const std::array<RotationLookup, rotationDegrees.size()>& rotationLookups()
    {
        static const std::array<RotationLookup, rotationDegrees.size()> lookups = {
            makeRotationLookup(rotationDegrees[0]),
            makeRotationLookup(rotationDegrees[1]),
            makeRotationLookup(rotationDegrees[2]),
            makeRotationLookup(rotationDegrees[3]),
            makeRotationLookup(rotationDegrees[4]),
            makeRotationLookup(rotationDegrees[5]),
            makeRotationLookup(rotationDegrees[6])
        };

        return lookups;
    }

    std::size_t nearestRotationIndexForDegrees(int degrees)
    {
        std::size_t bestIndex = 0;
        int bestDistance = std::abs(degrees - rotationDegrees[0]);

        for(std::size_t i = 1; i < rotationDegrees.size(); i++)
        {
            const int distance = std::abs(degrees - rotationDegrees[i]);
            if(distance < bestDistance)
            {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    std::size_t nearestRotationIndexForRadians(Scalar theta)
    {
        const int degrees = static_cast<int>(std::round(theta * 180.0f / pi));
        return nearestRotationIndexForDegrees(degrees);
    }

    std::vector<Scalar> rotateExampleWithLookup(const std::vector<Scalar>& input,
                                                std::size_t sizeX,
                                                std::size_t sizeY,
                                                std::size_t lookupIndex)
    {
        if(input.size() != sizeX * sizeY)
            throw std::runtime_error("Input doesn't match requested image size");

        if(sizeX != rotationImageSide || sizeY != rotationImageSide)
            throw std::runtime_error("Precomputed rotation lookup only supports 28x28 images");

        std::vector<Scalar> output(rotationImageSize, 0.0f);
        const RotationLookup& lookup = rotationLookups()[lookupIndex];

        for(std::size_t sourceIndex = 0; sourceIndex < rotationImageSize; sourceIndex++)
        {
            const Scalar sourceValue = input[sourceIndex];
            const SourcePixelRotation& sourcePixel = lookup.sourcePixels[sourceIndex];

            for(std::size_t i = 0; i < sourcePixel.count; i++)
            {
                const RotationContribution& contribution = sourcePixel.contributions[i];
                output[contribution.targetIndex] += sourceValue * contribution.weight;
            }
        }

        return output;
    }
}

CNN::CNN()
: convLayer_{ConvLayer(8, 1, 28, 28), ConvLayer(16, 8, 14, 14), ConvLayer(32, 16, 7, 7)},
  classifier_({288,128,10}, ActivationType::Relu, ActivationType::Softmax, 0.1f),
  nLayers_(3)
{
}

void CNN::configure(std::vector<std::size_t>convOutputChannels,
                    std::size_t classifierHiddenLayerSize,
                    Scalar dropout)
{
    nLayers_ = convOutputChannels.size();

    convLayer_.clear();
    convLayer_.reserve(nLayers_);

    if(nLayers_ == 0)
        throw std::runtime_error("CNN requires at least one convolution layer");

    std::size_t inChannels = 1;
    std::size_t height = 28;
    std::size_t width = 28;

    for(std::size_t outChannels : convOutputChannels)
    {
        outChannels = std::max<std::size_t>(outChannels, 1);
        convLayer_.emplace_back(outChannels, inChannels, height, width);
        inChannels = outChannels;
        height = convLayer_.back().getOutputHeight();
        width = convLayer_.back().getOutputWidth();
    }

    classifierHiddenLayerSize = std::max<std::size_t>(classifierHiddenLayerSize, 1);
    dropout = std::clamp<Scalar>(dropout, 0.0f, 0.95f);

    const ConvLayer& finalLayer = convLayer_.back();
    const std::size_t classifierInputs = finalLayer.getOutputChannels() * finalLayer.getOutputHeight() * finalLayer.getOutputWidth();
    std::cout << "Classifier inputs: " << classifierInputs << " output height: " << finalLayer.getOutputHeight() << " output width: " << finalLayer.getOutputWidth() << " nLayers: " << nLayers_ << std::endl;
    classifier_.configure(
        {static_cast<int>(classifierInputs), static_cast<int>(classifierHiddenLayerSize), 10},
        ActivationType::Relu,
        ActivationType::Softmax,
        dropout);
}

void CNN::print() const
{
    ;
}


const Tensor& CNN::forward(const Tensor& input)
{
    const Tensor* x = &convLayer_[0].forward(input);

    for(std::size_t i = 1; i < nLayers_; i++)
        x = &convLayer_[i].forward(*x);

    return *x;
}

void CNN::backward(const Tensor& outputGradient)
{
    Tensor x;

    x = convLayer_[nLayers_-1].backward(outputGradient, true);

    for(std::size_t i = nLayers_ - 1; i-- > 1;)
    {
            x = convLayer_[i].backward(x, true);
    }

    convLayer_[0].backward(x, false);
}


void CNN::gradient_descent(std::size_t trainingSize, double learningRate)
{
    //update Adam parameters
    ConvLayer::beta1pow *= ConvLayer::beta1;
    ConvLayer::beta2pow *= ConvLayer::beta2;

    for(std::size_t i = 0; i < nLayers_; i++)
        convLayer_[i].gradient_descent(trainingSize, learningRate);

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

    for(std::size_t i = 0; i < nLayers_; i++)
        convLayer_[i].zeroGradients();

    Matrix MLPinputs(classifier_.nInputs_(),batch.size());
    Matrix MLPtargets(classifier_.nOutputs_(),batch.size());

    //CNN forward pass and populate matrix of CNN outputs for entire batch
    for(std::size_t index=0; index<batch.size(); index++)
    {
        //load example into Tensor
        Tensor input({1,28,28});

        const std::size_t rotationIndex = static_cast<std::size_t>(rand()) % rotationDegrees.size();
        std::vector<Scalar> augmented = rotateExampleWithLookup(training_data.get_data()[batch[index]].inputs,28,28,rotationIndex);
        augmented = offsetExample(augmented, 28, 28, -4 + rand() % 8, -4 + rand () % 8);
        
        for(std::size_t i=0; i<training_data.n_inputs(); i++)
            input.data()[i] = augmented[i];

        for(std::size_t i=0; i<training_data.n_outputs(); i++)
            MLPtargets(i,index)=training_data.get_data()[batch[index]].outputs[i];

        //Put through CNN
        std::chrono::steady_clock::time_point forwardStart;
        if constexpr (profileCNN)
            forwardStart = std::chrono::steady_clock::now();

        const auto& outputTensor = forward(input);

        //cache network values for backwards pass
        for(std::size_t i = 0; i < nLayers_; i++)
            convLayer_[i].pushCache();

        if constexpr (profileCNN)
            forwardElapsed += std::chrono::steady_clock::now() - forwardStart;

        for(std::size_t j=0; j<outputTensor.size(); j++)
            MLPinputs(j,index) = outputTensor.data()[j];
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

    Tensor outputGradient({convLayer_[nLayers_-1].getOutputChannels(), convLayer_[nLayers_-1].getOutputHeight(), convLayer_[nLayers_-1].getOutputWidth()});

    for(std::size_t index=0; index<batch.size(); index++)
    {
        //retrieve cache values from forward run in order for backprop to work

        std::chrono::steady_clock::time_point forwardStart;
        if constexpr (profileCNN)
            forwardStart = std::chrono::steady_clock::now();

        for(std::size_t i = 0; i < nLayers_; i++)
            convLayer_[i].popCache();

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


std::pair<std::size_t, Scalar> CNN::predict(const std::vector<Scalar>& input)
{
    if((convLayer_[0].getInputWidth()*convLayer_[0].getInputHeight())!=input.size())throw std::runtime_error("Input doesn't match CNN shape");

    //load example into Tensor
    Tensor inputTensor({1,convLayer_[0].getInputHeight(),convLayer_[0].getInputWidth()});

    for(std::size_t j=0; j<input.size(); j++)
        inputTensor.data()[j] = input[j];

    //Put through CNN
    auto outputTensor = forward(inputTensor);

    auto vec = std::vector<Scalar>(outputTensor.data(),outputTensor.data()+outputTensor.size());
    set_inputMLP(vec);
    propagateMLP();

    return std::pair<std::size_t, Scalar>(find_highest_output(),get_output(find_highest_output()));
}


void CNN::save(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);

    if(!file)
        throw std::runtime_error("Save file could not be opened");

    //Magic number
    constexpr char MAGIC[] = {'N','C','N','N'};
    file.write(MAGIC, sizeof(MAGIC));

    //Version
    const uint8_t version = 4;
    file.write(reinterpret_cast<const char*>(&version),sizeof(version));

    //CNN Layers
    file.write(reinterpret_cast<const char*>(&nLayers_),sizeof(nLayers_));

    for(std::size_t i = 0; i < nLayers_; i++)
        convLayer_[i].save(file);

        classifier_.save(file);

    if(!file)
        throw std::runtime_error("Failed while saving");
}


void CNN::load(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if(!file)
        throw std::runtime_error("Load file could not be opened");

    char MAGIC[] = {'X','X','X','X'};
    file.read(reinterpret_cast<char*>(&MAGIC), sizeof(MAGIC));

    if(std::string(MAGIC,sizeof(MAGIC))!="NCNN")
        throw std::runtime_error("Not a valid CNN file");

    //Version
    std::uint8_t version;
    file.read(reinterpret_cast<char*>(&version),sizeof(version));

    if(version!=4)
        throw std::runtime_error("Unsupported file version");

    //CNN Layers
    file.read(reinterpret_cast<char*>(&nLayers_),sizeof(nLayers_));

    for(std::size_t i = 0; i < nLayers_; i++)
        convLayer_[i].load(file);

    classifier_.load(file);

    const std::size_t lastLayer = nLayers_ - 1;
    const std::size_t classifierInputs = convLayer_[lastLayer].getOutputChannels() * convLayer_[lastLayer].getOutputHeight() * convLayer_[lastLayer].getOutputWidth();
    if(classifier_.nInputs_() != classifierInputs)
        throw std::runtime_error("Loaded CNN classifier input size does not match convolution output size");

    if(!file)
        throw std::runtime_error("Failed while loading CNN");
}

std::vector<Scalar> CNN::offsetExample(const std::vector<Scalar>& input, std::size_t sizeX, std::size_t sizeY, int offsetX, int offSetY)
{
    std::vector<Scalar> output(sizeX * sizeY);


    for(std::size_t targetY = 0; targetY < sizeY; targetY++)
        for(std::size_t targetX = 0; targetX < sizeX; targetX++)
        {
            output.data()[targetY * sizeX + targetX] =
            (targetX + offsetX < 0 || targetX + offsetX > sizeX - 1 || targetY + offSetY < 0 || targetY + offSetY > sizeY - 1) ? 0.0f :
            input.data()[(targetY + offSetY) * sizeX + (targetX + offsetX)];
        }
    return output;
}


std::vector<Scalar> CNN::rotateExample(const std::vector<Scalar>& input, std::size_t sizeX, std::size_t sizeY, Scalar theta)
{
    return rotateExampleWithLookup(input, sizeX, sizeY, nearestRotationIndexForRadians(theta));
}
