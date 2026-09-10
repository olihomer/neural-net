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
}


AppEngine::AppEngine()
: net({784,128,10}, ActivationType::Relu, ActivationType::Softmax)
{
    std::cout << "Constructing Engine" << std::endl;
}


int AppEngine::runApp(void(*progress)(int32_t,double), int hiddenLayerSize, int epochs, int trainingExamples, int batchSize, double learningRate, int hiddenActivation, int outputActivation)
{
    hiddenLayerSize = std::max(hiddenLayerSize, 1);
    epochs = std::max(epochs, 1);
    trainingExamples = std::max(trainingExamples, 1);
    batchSize = std::max(batchSize, 1);
    learningRate = std::max(learningRate, 0.0);

    const ActivationType hiddenActivationType = activationTypeFor(hiddenActivation);
    const ActivationType outputActivationType = activationTypeFor(outputActivation);
    
    net.configure({784, hiddenLayerSize, 10}, hiddenActivationType, outputActivationType);
    
    //train network
    
    Trainer trainer(net);
   
    mnist_data mnist_training_data("/Users/oliverhomer/Xcode/neural net v1/mnist_test.csv", trainingExamples);
    
    trainer.train(mnist_training_data, epochs, batchSize, learningRate, progress);
    
    //quick check
    
    mnist_data mnist_training_data2("/Users/oliverhomer/Xcode/neural net v1/mnist_test.csv", trainingExamples);

    int wrong = 0;
    const int evaluationExamples = std::min(trainingExamples, 100);
    
    for(int i=0;i<evaluationExamples;i++)
    {
        int guess = i;
        int guess_label = mnist_training_data2.get_label(guess);
        std::cout << "Guess = " << guess_label;
        
        net.set_input(mnist_training_data2, guess);
        net.propagate();
        
        std::cout << ". Net guessed " << net.find_highest_output() << " with value of " << net.get_output(net.find_highest_output()) << std::endl;
        if(net.find_highest_output()!=guess_label){std::cout<<"WRONG!"<<std::endl;wrong++;}
    }
    
    std::cout << "Success rate: " << (1 - (float(wrong) / float(evaluationExamples)) ) << std::endl;
    
    return 0;
}


bool AppEngine::saveNetwork(const char *path)
{
    if(path == nullptr)return false;

    try
    {
        net.save(path);
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
        net.load(path);
        return true;
    }
    catch(const std::exception& error)
    {
        std::cout << "Load failed: " << error.what() << std::endl;
        return false;
    }
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

    return net.predict(vectorData);;
     
}


