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
#include "ImageAugmenter.hpp"
#include "GradientView.hpp"


namespace
{
    constexpr std::size_t FILE_VERSION = 6;
}

CNN::CNN()
: convLayer_{ConvLayer(8, 1, 28, 28,true), ConvLayer(16, 8, 14, 14,true), ConvLayer(32, 16, 7, 7,false)},
  classifier_({1568,128,10}, ActivationType::Relu, ActivationType::Softmax, 0.1f),
  nLayers_(3)
{
}

void CNN::configure(std::vector<std::size_t>convOutputChannels,
                    std::vector<bool>bPoolingPerLayer,
                    std::size_t classifierHiddenLayerSize,
                    Scalar dropout,
                    std::size_t mlpOutputs,
                    std::size_t inputX,
                    std::size_t inputY,
                    std::size_t inputChannels)
{
    nLayers_ = convOutputChannels.size();

    convLayer_.clear();
    convLayer_.reserve(nLayers_);

    if(nLayers_ == 0)
        throw std::runtime_error("CNN requires at least one convolution layer");

    std::size_t inChannels = inputChannels;
    std::size_t height = inputY;
    std::size_t width = inputX;

    std::size_t index = 0;
    
    for(std::size_t outChannels : convOutputChannels)
    {
        outChannels = std::max<std::size_t>(outChannels, 1);
        const bool poolingEnabled = index < bPoolingPerLayer.size() ? bPoolingPerLayer[index] : false;
        convLayer_.emplace_back(outChannels, inChannels, height, width, poolingEnabled);
        inChannels = outChannels;
        height = convLayer_.back().getOutputHeight();
        width = convLayer_.back().getOutputWidth();
        index++;
    }

    classifierHiddenLayerSize = std::max<std::size_t>(classifierHiddenLayerSize, 1);
    dropout = std::clamp<Scalar>(dropout, 0.0f, 0.95f);

    const ConvLayer& finalLayer = convLayer_.back();
    const std::size_t classifierInputs = finalLayer.getOutputChannels() * finalLayer.getOutputHeight() * finalLayer.getOutputWidth();
    std::cout << "Classifier inputs: " << classifierInputs << " output height: " << finalLayer.getOutputHeight() << " output width: " << finalLayer.getOutputWidth() << " nLayers: " << nLayers_ << std::endl;
    classifier_.configure(
        {static_cast<int>(classifierInputs), static_cast<int>(classifierHiddenLayerSize), static_cast<int>(mlpOutputs)},
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

const Tensor& CNN::forwardBatch(const Tensor& input)
{
    //std::cout << "Forwarding to ConvLayer 0" << std::endl;
    const Tensor* x = &convLayer_[0].forwardBatch(input);

    for(std::size_t i = 1; i < nLayers_; i++)
    {
        //std::cout << "Forwarding to ConvLayer" << i << std::endl;
        x = &convLayer_[i].forwardBatch(*x);
    }
        
    return *x;
}



void CNN::backward(const Tensor& outputGradient)
{
    Tensor x;

    if(nLayers_>1)x = convLayer_[nLayers_-1].backward(outputGradient, true);

    for(std::size_t i = nLayers_ - 1; i-- > 1;)
    {
            x = convLayer_[i].backward(x, true);
    }

    convLayer_[0].backward(x, false);
}

void CNN::backwardBatch(const GradientView& outputGradient, std::size_t miniBatchIndex)
{
    auto x = convLayer_[nLayers_-1].backwardBatch(outputGradient, true, (int)miniBatchIndex);

    for(std::size_t i = nLayers_ - 1; i-- > 1;)
    {
            x = convLayer_[i].backwardBatch(x, true, (int)miniBatchIndex);
    }

    convLayer_[0].backwardBatch(x, false, (int)miniBatchIndex);
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

double CNN::trainBatch(const DataSet& training_data, const std::span<const std::size_t> batch)
{
    double total_error = 0;
    
    for(std::size_t i = 0; i < nLayers_; i++)
        convLayer_[i].zeroGradients();
    
    Matrix MLPinputs(classifier_.nInputs_(),batch.size());
    Matrix MLPtargets(classifier_.nOutputs_(),batch.size());
    ImageAugmenter augmenter;
    std::vector<Scalar> rotated(training_data.n_inputs());
    std::vector<Scalar> augmented(training_data.n_inputs());
    
    //batch processing for CNN convolution
    std::size_t batchIndex = 0;
    std::size_t miniBatchSize = convLayer_[0].miniBatchSize;
    
    Tensor input({miniBatchSize,convLayer_[0].inputChannels(),convLayer_[0].inputHeight(),convLayer_[0].inputWidth()});
    const std::size_t inputSize = convLayer_[0].inputChannels() * convLayer_[0].inputHeight() * convLayer_[0].inputWidth();
    
    //populate the input tensor with the miniBatch
    
    if(training_data.n_inputs() != inputSize)
        throw std::runtime_error("Dataset and CNN input shapes do not match.");
    
    while(batchIndex < batch.size())
    {
        input.zero();
        std::size_t thisMiniBatchSize = 0;
        for(std::size_t miniBatchIndex = 0; miniBatchIndex < miniBatchSize; miniBatchIndex++)
        {
            Scalar* inputBatch = input.data() + miniBatchIndex * inputSize;
            
            
            //const std::size_t rotationIndex = static_cast<std::size_t>(rand()) % augmenter.rotationCount();
            //augmenter.rotate(training_data.get_data()[batch[batchIndex]].inputs.data(), rotated.data(), 28, 28, rotationIndex);
            //augmenter.translate(rotated.data(), augmented.data(), 28, 28, -2 + rand() % 5, -2 + rand() % 5);

            const std::vector<Scalar>& source = training_data.get_data()[batch[batchIndex]].inputs;
            const Scalar* cropSource = source.data();

                if(rand() % 2)
                {
                    augmenter.flip(source.data(),augmented.data(), convLayer_[0].inputWidth(), convLayer_[0].inputHeight(), convLayer_[0].inputChannels());
                    cropSource = augmented.data();
                }
            
            augmenter.randomCrop(cropSource, inputBatch, convLayer_[0].inputWidth(), convLayer_[0].inputHeight(), convLayer_[0].inputChannels());
                
            //populate the matching targets
            for(std::size_t i=0; i<training_data.n_outputs(); i++)
                MLPtargets(i,batchIndex) = training_data.get_data()[batch[batchIndex]].outputs[i];
            
            batchIndex++;
            thisMiniBatchSize++;
            if(batchIndex == batch.size())break;
        }
        
        //Now convolve the minibatch of inputs
        const auto& outputTensor = forwardBatch(input);
        
        //we've now got outputTensor which is 784 outputs x 16 minibatches, arranged into 16 columns
        //let's copy it into MLPInput
        
        std::size_t inputsToCopy = outputTensor.size()/miniBatchSize;
        const std::size_t batchStart = batchIndex - thisMiniBatchSize;
        
        for(std::size_t local = 0; local < thisMiniBatchSize; local++)
        {
            const std::size_t global = batchStart + local;
            
            for(std::size_t j = 0; j < inputsToCopy; j++)
                MLPinputs(j,global) = outputTensor.data()[local * inputsToCopy + j];
        }
        
        //cache network values for backwards pass
        for(std::size_t i = 0; i < nLayers_; i++)
            convLayer_[i].pushCacheBatch(thisMiniBatchSize);
        
        
    }
    
    //Run outputs from CNN through the MLP
    
    total_error = classifier_.trainBatch(MLPinputs, MLPtargets);
    
    Matrix inputError = classifier_.get_input_error();
    
    //inputError matrix now contains error gradients for entire batch. Backprop through CNN one minibatch at a time
    
    Tensor outputGradient({miniBatchSize, convLayer_[nLayers_-1].getOutputChannels(), convLayer_[nLayers_-1].getOutputHeight(), convLayer_[nLayers_-1].getOutputWidth()});
    
    //Batch backprop through MLP
    
    std::size_t batchStartIndex = 0;
    
    while(convLayer_[0].cacheSize > 0) // walk through the cache minibatches
    {
        std::size_t thisMiniBatchSize = 0;
        
        for(std::size_t i = 0; i < nLayers_; i++)
            thisMiniBatchSize = convLayer_[i].popCacheBatch(); //pop network values from the cache
        
        GradientView gradient(inputError.data() + batchStartIndex, inputError.rows() * thisMiniBatchSize, 1, inputError.cols());
        
        backwardBatch(gradient, thisMiniBatchSize);
        
        batchStartIndex += thisMiniBatchSize;
    }
    
    return total_error;
}

void CNN::print_stats(std::ostream& ostream)
{
    ;
}


std::pair<std::size_t, Scalar> CNN::predict(const std::vector<Scalar>& input)
{
    const auto channels = convLayer_[0].inputChannels();
    const auto height = convLayer_[0].inputHeight();
    const auto width = convLayer_[0].inputWidth();
    
    if(channels*height*width!=input.size())throw std::runtime_error("Input doesn't match CNN shape");
    
    //load example into Tensor
    Tensor inputTensor({channels,height,width});

    std::copy(input.begin(),input.end(), inputTensor.data());
    
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
    const uint8_t version = FILE_VERSION;
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

    if(version!=FILE_VERSION)
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
