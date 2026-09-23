//
//  AppEngine.cpp
//  neural net v1
//
//  Created by Oliver Homer on 18/02/2024.
//

#include <iostream>
#include <fstream>
#include "AppEngine.hpp"
#include "Neural.hpp"
#include "data_set.hpp"
#include "mnist_data.hpp"
#include "ActivationFunction.hpp"
#include "Trainer.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <numeric>
#include "Matrix.hpp"
#include "Tensor.hpp"
#include "CNN.hpp"


namespace {

const ActivationType activationTypeFor(int choice)
{
    switch(choice)
    {
        case 0:
            return ActivationType::Relu;
        case 1:
            return ActivationType::Sigmoid;
        case 2:
            return ActivationType::Softmax;
        default:
            return ActivationType::Sigmoid;
    }
}

const ConvLayer& cnnLayerFor(const CNN& cnn, int layer)
{
    return layer == 2 ? cnn.conv2_ : cnn.conv1_;
}
}


AppEngine::AppEngine()
: mlp_({784,128,10}, ActivationType::Relu, ActivationType::Softmax)
{
    std::cout << "Constructing Engine" << std::endl;
    
}


void AppEngine::selectModel(ModelKind kind)
{
    activeModel_ = kind;
}


int AppEngine::runApp(void(*progress)(int32_t,double),
                      int modelKind,
                      int hiddenLayerSize,
                      int epochs,
                      int trainingExamples,
                      int batchSize,
                      double learningRate,
                      int hiddenActivation,
                      int outputActivation,
                      int cnnConv1Channels,
                      int cnnConv2Channels,
                      int cnnClassifierHiddenLayerSize)
{
    std::cout << "Entered RunApp" << std::endl;
    modelKind = std::clamp(modelKind, 0, 1);
    hiddenLayerSize = std::max(hiddenLayerSize, 1);
    epochs = std::max(epochs, 1);
    trainingExamples = std::max(trainingExamples, 1);
    batchSize = std::max(batchSize, 1);
    learningRate = std::max(learningRate, 0.0);
    cnnConv1Channels = std::max(cnnConv1Channels, 1);
    cnnConv2Channels = std::max(cnnConv2Channels, 1);
    cnnClassifierHiddenLayerSize = std::max(cnnClassifierHiddenLayerSize, 1);

    const ActivationType hiddenActivationType = activationTypeFor(hiddenActivation);
    const ActivationType outputActivationType = activationTypeFor(outputActivation);
    const ModelKind selectedModel = modelKind == static_cast<int>(ModelKind::CNN) ? ModelKind::CNN : ModelKind::MLP;
    
    const int evaluationExamples = 1000;
    
    mnist_data mnist("/Users/oliverhomer/Xcode/neural net v1/mnist_test.csv", trainingExamples);
    
    selectModel(selectedModel);

    if(activeModel_ == ModelKind::CNN)
    {
        cnn_.configure(
            static_cast<std::size_t>(cnnConv1Channels),
            static_cast<std::size_t>(cnnConv2Channels),
            static_cast<std::size_t>(cnnClassifierHiddenLayerSize));
    }
    else
    {
        mlp_.configure({784, hiddenLayerSize, 10}, hiddenActivationType, outputActivationType);
    }

    Trainer trainer(activeTrainable());
    trainer.train(mnist, epochs, batchSize, learningRate, progress);
    
    mnist_data mnist_training_data2("/Users/oliverhomer/Xcode/neural net v1/mnist_test.csv", trainingExamples + evaluationExamples);

    int wrong = 0;
    
    for(int i=0;i<evaluationExamples;i++)
    {
        int guess = i+trainingExamples;
        int guess_label = mnist_training_data2.get_label(guess);
        //std::cout << "Guess = " << guess_label;
        
        std::pair<std::size_t, Scalar> prediction =
            activeModel_ == ModelKind::CNN
            ? cnn_.predict(mnist_training_data2.get_data()[guess].inputs)
            : mlp_.predict(mnist_training_data2.get_data()[guess].inputs);
        
        //std::cout << ". Net guessed " << prediction.first << " with value of " << prediction.second << std::endl;
        if(prediction.first!=guess_label){/*std::cout<<"WRONG!"<<std::endl;*/wrong++;}
    }
    
    std::cout << "Success rate: " << (1 - (float(wrong) / float(evaluationExamples)) ) << std::endl;
    
    return 0;
}


bool AppEngine::saveNetwork(const char *path)
{
    if(path == nullptr)return false;

    try
    {
        activeModel_ == ModelKind::CNN
        ? cnn_.save(path)
        : mlp_.save(path);
        return true;
    }
    catch(const std::exception& error)
    {
        std::cout << "Save failed: " << error.what() << std::endl;
        return false;
    }
}


bool AppEngine::loadNetwork(const char *path)
{
    if(path == nullptr)return false;

    try
    {
        activeModel_ == ModelKind::CNN
        ? cnn_.load(path)
        : mlp_.load(path);
        return true;
    }
    catch(const std::exception& error)
    {
        std::cout << "Load failed: " << error.what() << std::endl;
        return false;
    }
}


int AppEngine::activeModelKind() const
{
    return static_cast<int>(activeModel_);
}


int AppEngine::cnnKernelOutputChannels(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).kernelOutputChannels());
}


int AppEngine::cnnKernelInputChannels(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).kernelInputChannels());
}


int AppEngine::cnnKernelHeight(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).kernelHeight());
}


int AppEngine::cnnKernelWidth(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).kernelWidth());
}


float AppEngine::cnnKernelValue(int layer, int outputChannel, int inputChannel, int y, int x) const
{
    if(outputChannel < 0 || inputChannel < 0 || y < 0 || x < 0)
        return 0.0f;

    const ConvLayer& convLayer = cnnLayerFor(cnn_, layer);
    const auto out = static_cast<std::size_t>(outputChannel);
    const auto in = static_cast<std::size_t>(inputChannel);
    const auto row = static_cast<std::size_t>(y);
    const auto col = static_cast<std::size_t>(x);

    if(out >= convLayer.kernelOutputChannels() ||
       in >= convLayer.kernelInputChannels() ||
       row >= convLayer.kernelHeight() ||
       col >= convLayer.kernelWidth())
    {
        return 0.0f;
    }

    return convLayer.kernelValue(out, in, row, col);
}


int AppEngine::cnnInputChannels(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).inputChannels());
}


int AppEngine::cnnInputHeight(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).inputHeight());
}


int AppEngine::cnnInputWidth(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).inputWidth());
}


float AppEngine::cnnInputValue(int layer, int channel, int y, int x) const
{
    if(channel < 0 || y < 0 || x < 0)
        return 0.0f;

    const ConvLayer& convLayer = cnnLayerFor(cnn_, layer);
    const auto inputChannel = static_cast<std::size_t>(channel);
    const auto row = static_cast<std::size_t>(y);
    const auto col = static_cast<std::size_t>(x);

    if(inputChannel >= convLayer.inputChannels() ||
       row >= convLayer.inputHeight() ||
       col >= convLayer.inputWidth())
    {
        return 0.0f;
    }

    return convLayer.inputValue(inputChannel, row, col);
}


int AppEngine::cnnActivationChannels(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).activationChannels());
}


int AppEngine::cnnActivationHeight(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).activationHeight());
}


int AppEngine::cnnActivationWidth(int layer) const
{
    return static_cast<int>(cnnLayerFor(cnn_, layer).activationWidth());
}


float AppEngine::cnnActivationValue(int layer, int channel, int y, int x) const
{
    if(channel < 0 || y < 0 || x < 0)
        return 0.0f;

    const ConvLayer& convLayer = cnnLayerFor(cnn_, layer);
    const auto activationChannel = static_cast<std::size_t>(channel);
    const auto row = static_cast<std::size_t>(y);
    const auto col = static_cast<std::size_t>(x);

    if(activationChannel >= convLayer.activationChannels() ||
       row >= convLayer.activationHeight() ||
       col >= convLayer.activationWidth())
    {
        return 0.0f;
    }

    return convLayer.activationValue(activationChannel, row, col);
}


std::pair<int,float> AppEngine::sendRasterData(const float *data, std::size_t size)
{
    if(data==nullptr)return std::pair<int,float>(0,0);

    const std::size_t side = static_cast<std::size_t>(std::sqrt(size));
    if(side * side != size)
    {
        std::cout << "Error: raster size " << size << " is not square" << std::endl;
        return std::pair<int,float>(0,0);
    }

    std::vector<float> vectorData(data, data + size);
    
    vectorData = mnist_data::preProcess(vectorData);
    
    std::cout << "Raster after processing:" << std::endl;
    
    for(std::size_t y = 0; y < 28; y++)
    {
        for(std::size_t x = 0; x < 28; x++)
            std::cout << (vectorData[x+y*28] > 50.0f/255.0f ? "X " : "  ");
        std::cout << std::endl;
    }

    const auto prediction = activeModel_ == ModelKind::CNN ? cnn_.predict(vectorData) : mlp_.predict(vectorData);
    return std::pair<int,float>(static_cast<int>(prediction.first), prediction.second);
     
}


Trainable& AppEngine::activeTrainable()
{
    if(activeModel_ == ModelKind::CNN)
    {
        return cnn_;
    }

    return mlp_;
};
