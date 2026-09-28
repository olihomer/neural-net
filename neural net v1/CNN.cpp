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

namespace
{
    constexpr bool profileCNN = false;
}

CNN::CNN()
:convLayer_{ConvLayer(8, 1, 28, 28), ConvLayer(16, 8, 14, 14)},classifier_({784,128,10}, ActivationType::Relu, ActivationType::Softmax, 0.1f), nLayers_(2)
{
    
}

void CNN::configure(std::vector<std::size_t>convOutputChannels,
                    std::size_t classifierHiddenLayerSize,
                    Scalar dropout)
{
    nLayers_ = std::min<std::size_t>(convOutputChannels.size(), MAX_LAYERS);
    if(nLayers_ == 0)
        throw std::runtime_error("CNN requires at least one convolution layer");

    for(std:: size_t i = 0; i < nLayers_; i++)
    {
        convOutputChannels[i] = std::max<std::size_t>(convOutputChannels[i], 1);
        convLayer_[i] = ConvLayer(convOutputChannels[i], i==0 ? 1 : convOutputChannels[i-1], 28/std::pow(2,i), 28/std::pow(2,i));
    }
        
    classifierHiddenLayerSize = std::max<std::size_t>(classifierHiddenLayerSize, 1);
    dropout = std::clamp<Scalar>(dropout, 0.0f, 0.95f);

    const std::size_t finalSide = convLayer_[nLayers_ - 1].getOutputHeight();
    const std::size_t classifierInputs = convOutputChannels[nLayers_ - 1] * finalSide * finalSide;
    classifier_.configure(
        {static_cast<int>(classifierInputs), static_cast<int>(classifierHiddenLayerSize), 10},
        ActivationType::Relu,
        ActivationType::Softmax,
        dropout);
    
    std::cout << "Configuring CNN with convOutputChannels: " << convOutputChannels[0] << "," << convOutputChannels[1] << " and nLayers: " << nLayers_ << std::endl;
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

        int offsetX = std::round(((Scalar)rand()/RAND_MAX) * 4.0f) - 2;
        int offsetY = std::round(((Scalar)rand()/RAND_MAX) * 4.0f) - 2;
        
        std::vector<Scalar> offset = CNN::offsetExample(training_data.get_data()[batch[index]].inputs,28,28,offsetX,offsetY);
        
        for(std::size_t i=0; i<training_data.n_inputs(); i++)
            input.data()[i]= offset[i];

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
    if(nLayers_ == 0 || nLayers_ > MAX_LAYERS)
        throw std::runtime_error("Unsupported CNN layer count");

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
