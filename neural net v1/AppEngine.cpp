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
#include "DataSet.hpp"
#include "MnistData.hpp"
#include "ActivationFunction.hpp"
#include "Trainer.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <numeric>
#include <array>
#include "Matrix.hpp"
#include "Tensor.hpp"
#include "CNN.hpp"
#include "ImageAugmenter.hpp"
#include "EmnistData.hpp"
#include "Cifar10.hpp"

namespace {

constexpr Scalar pi = 3.14159265358979323846f;
constexpr std::array<int, 7> visualizationRotationDegrees = {-15, -10, -5, 0, 5, 10, 15};

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
    const int layerIndex = std::clamp(layer - 1, 0, static_cast<int>(cnn.convLayer_.size()) - 1);
    return cnn.convLayer_[static_cast<std::size_t>(layerIndex)];
}
}


AppEngine::AppEngine()
: mlp_({784,128,10}, ActivationType::Relu, ActivationType::Softmax, 0.1f)
{
    std::cout << "Constructing Engine" << std::endl;
 
    try
    {
        Cifar10("/Users/oliverhomer/Xcode/neural net v1/data/data_batch_1", 10);
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        exit(99);
    }
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
                      int evaluationExamples,
                      int batchSize,
                      double learningRate,
                      int hiddenActivation,
                      int outputActivation,
                      const int *cnnConvChannels,
                      int cnnConvLayerCount,
                      int cnnClassifierHiddenLayerSize,
                      double dropout)
{
    std::cout << "Entered RunApp" << std::endl;
    modelKind = std::clamp(modelKind, 0, 1);
    hiddenLayerSize = std::max(hiddenLayerSize, 1);
    epochs = std::max(epochs, 1);
    trainingExamples = std::max(trainingExamples, 1);
    batchSize = std::max(batchSize, 1);
    learningRate = std::max(learningRate, 0.0);
    cnnClassifierHiddenLayerSize = std::max(cnnClassifierHiddenLayerSize, 1);
    dropout = std::clamp(dropout, 0.0, 0.95);

    std::vector<std::size_t> cnnLayerChannels;
    if(cnnConvChannels != nullptr && cnnConvLayerCount > 0)
    {
        cnnLayerChannels.reserve(static_cast<std::size_t>(cnnConvLayerCount));
        for(int i = 0; i < cnnConvLayerCount; i++)
            cnnLayerChannels.push_back(static_cast<std::size_t>(std::max(cnnConvChannels[i], 1)));
    }

    if(cnnLayerChannels.empty())
        cnnLayerChannels = {8, 16, 32};

    const ActivationType hiddenActivationType = activationTypeFor(hiddenActivation);
    const ActivationType outputActivationType = activationTypeFor(outputActivation);
    const ModelKind selectedModel = modelKind == static_cast<int>(ModelKind::CNN) ? ModelKind::CNN : ModelKind::MLP;
        
    EmnistData emnistDataSet("/Users/oliverhomer/Xcode/neural net v1/data/emnist-balanced-train", trainingExamples);

    selectModel(selectedModel);

    if(activeModel_ == ModelKind::CNN)
    {
        cnn_.configure(
            cnnLayerChannels,
            static_cast<std::size_t>(cnnClassifierHiddenLayerSize),
            static_cast<Scalar>(dropout),
            (int)emnistDataSet.n_outputs());
    }
    else
    {
        mlp_.configure({784, hiddenLayerSize, (int)emnistDataSet.n_outputs()}, hiddenActivationType, outputActivationType, static_cast<Scalar>(dropout));
    }

    Trainer trainer(activeTrainable());
    
    try
    {
        trainer.train(emnistDataSet, epochs, batchSize, learningRate, progress);
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 0;
    }
    
    EmnistData emnistDataSet2("/Users/oliverhomer/Xcode/neural net v1/data/emnist-balanced-test", evaluationExamples);

    int wrong = 0;
    
    Matrix confusionMatrix(emnistDataSet.n_outputs(),emnistDataSet.n_outputs());
    
    for(int i=0;i<evaluationExamples;i++)
    {
        int guess = i;
        int guess_label = emnistDataSet2.get_label(guess);
        //std::cout << "Guess = " << guess_label;
        
        std::pair<std::size_t, Scalar> prediction =
            activeModel_ == ModelKind::CNN
            ? cnn_.predict(emnistDataSet2.get_data()[guess].inputs)
            : mlp_.predict(emnistDataSet2.get_data()[guess].inputs);
        
        //std::cout << ". Net guessed " << prediction.first << " with value of " << prediction.second << std::endl;
        if(prediction.first!=guess_label)
        {
            std::cout << ". Net guessed " << EmnistData::labelSet[prediction.first] << " with confidence of " << prediction.second << ". Correct label was " << EmnistData::labelSet[guess_label] << std::endl;
            wrong++;
            confusionMatrix(prediction.first,guess_label)++;
        }
    }

    //print confusion matrix
    
    std::cout <<  "  ";

    for(std::size_t i = 0; i < emnistDataSet.n_outputs(); i++)
        std::cout << EmnistData::labelSet[i] << " ";
    std::cout << std::endl;
    
    for(std::size_t i = 0; i < emnistDataSet.n_outputs(); i++)
    {
        std::cout << EmnistData::labelSet[i] << " ";
        for(std::size_t j = 0; j < emnistDataSet.n_outputs(); j++)
            std::cout << confusionMatrix(i,j) << " ";
        std::cout << std::endl;
    }
        
    
    
    
    std::cout << "Success rate: " << (1 - (float(wrong) / float(evaluationExamples)) ) << std::endl;

    if(activeModel_ == ModelKind::CNN && trainingExamples > 0)
    {
        const std::size_t rotationIndex = static_cast<std::size_t>(rand()) % visualizationRotationDegrees.size();
        const Scalar theta = static_cast<Scalar>(visualizationRotationDegrees[rotationIndex]) * pi / 180.0f;
        ImageAugmenter augmenter;
        std::vector<Scalar> augmented(emnistDataSet.get_data()[0].inputs.size());
        augmenter.rotate(emnistDataSet.get_data()[0].inputs.data(), augmented.data(), 28, 28, theta);
        cnn_.predict(augmented);
    }
    
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


int AppEngine::cnnLayerCount() const
{
    return static_cast<int>(cnn_.convLayer_.size());
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
    
    vectorData = MnistData::preProcess(vectorData);
    
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
