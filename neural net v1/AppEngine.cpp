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
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <numeric>

namespace {
Relu reluActivationFunction;
Sigmoid sigmoidActivationFunction;
Softmax softmaxActivationFunction;

const ActivationFunction& activationFunctionFor(int choice)
{
    switch(choice)
    {
        case 0:
            return reluActivationFunction;
        case 1:
            return sigmoidActivationFunction;
        case 2:
            return softmaxActivationFunction;
        default:
            return sigmoidActivationFunction;
    }
}
}

AppEngine::AppEngine()
: net({784,128,10}, reluActivationFunction, softmaxActivationFunction)
{
    std::cout << "Constructing Engine" << std::endl;
}

AppEngine::~AppEngine()
{
    std::cout << "Deconstructing Engine" << std::endl;
}

int AppEngine::runApp(void(*progress)(int32_t,double), int hiddenLayerSize, int epochs, int trainingExamples, double learningRate, int hiddenActivation, int outputActivation)
{
    hiddenLayerSize = std::max(hiddenLayerSize, 1);
    epochs = std::max(epochs, 1);
    trainingExamples = std::max(trainingExamples, 1);
    learningRate = std::max(learningRate, 0.0);

    const ActivationFunction& hiddenActivationFunction = activationFunctionFor(hiddenActivation);
    const ActivationFunction& outputActivationFunction = activationFunctionFor(outputActivation);
    net.configure({784, hiddenLayerSize, 10}, hiddenActivationFunction, outputActivationFunction);

    mnist_data mnist_training_data("/Users/oliverhomer/Xcode/neural net v1/mnist_test.csv", trainingExamples);
    
    const std::size_t batchSize = 50;
    const std::size_t trainingExampleCount = static_cast<std::size_t>(trainingExamples);
    double total_error = 0;
 
    for(std::size_t i = 0; i < epochs; i++)
    {
        double epoch_error = 0;
        std::size_t batches = 0;

        for(std::size_t training_index = 0; training_index < trainingExampleCount; training_index += batchSize)
        {
            const std::size_t currentBatchSize = std::min(batchSize, trainingExampleCount - training_index);
            epoch_error += net.train(mnist_training_data, currentBatchSize, training_index);
            net.gradient_descent(currentBatchSize, learningRate);
            batches++;
        }

        total_error = epoch_error / double(batches);

        if((i+1) % 50 == 0 || i == epochs - 1)
        {
            std::cout << i << " ";
            std::cout << "Total error: " << total_error << std::endl;
            net.print_stats(std::cout);
            progress(int(i),total_error);
        }
        mnist_training_data.shuffle();
    }
    
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
    
    std::cout << "Success rate: " << (1 - (float(wrong) / float(evaluationExamples)) );
    
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

    std::vector<float> vectorData;
    vectorData.resize(size);
    
    for(std::size_t i = 0; i < size; i++)
        vectorData[i] = data[i];
    
    /*std::cout << "Raster with side " << side << " before processing:" << std::endl;
    
    for(std::size_t y = 0; y < side; y++)
    {
        for(std::size_t x = 0; x < side; x++)
            std::cout << (vectorData[x + y * side] > 50.0f/255.0f ? "X" : " ");
        std::cout << std::endl;
    }*/
    
    vectorData = preProcess(vectorData);
    
    std::cout << "Raster after processing:" << std::endl;
    
    for(std::size_t y = 0; y < 28; y++)
    {
        for(std::size_t x = 0; x < 28; x++)
            std::cout << (vectorData[x+y*28] > 50.0f/255.0f ? "X " : "  ");
        std::cout << std::endl;
    }
    
    
    net.set_input(vectorData);
    net.propagate();
    
    std::cout << ". Net guessed " << net.find_highest_output() << " with value of " << net.get_output(net.find_highest_output()) << std::endl;
    
    return std::pair<int,float>(net.find_highest_output(),net.get_output(net.find_highest_output()));
     
}


std::vector<float> AppEngine::preProcess(std::vector<float> input)
{
    std::vector<float> output;
    output.resize(28 * 28);
    
    const std::size_t sizeofside = static_cast<std::size_t>(std::sqrt(input.size()));
    if(input.empty() || sizeofside * sizeofside != input.size())
    {
        std::cout << "Error: input raster size " << input.size() << " is not square" << std::endl;
        return output;
    }
    
    std::size_t minX = sizeofside - 1;
    std::size_t maxX = 0;
    std::size_t minY = sizeofside - 1;
    std::size_t maxY = 0;
    
    const float threshold = 5.0f/255.0f;
    
    //establish bounding box and centre of mass
    
    float weighted_x = 0;
    float weighted_y = 0;
    float total_weight = 0;
    
    for(std::size_t y = 0; y < sizeofside; y++)
        for(std::size_t x = 0; x < sizeofside; x++)
        {
            float pixel = input[y*sizeofside+x];
            weighted_x += x * pixel;
            weighted_y += y * pixel;
            total_weight += pixel;
            
            if(pixel > threshold)
            {
                minX = x < minX ? x : minX;
                minY = y < minY ? y : minY;
                maxX = x > maxX ? x : maxX;
                maxY = y > maxY ? y : maxY;
            }
        }
    
    std::cout << "X range: " << minX << " to " << maxX << std::endl;
    std::cout << "Y range: " << minY << " to " << maxY << std::endl;
    
    //Blank drawing, return blank
    if (minX > maxX || minY > maxY) return output;
    
    //establish dimensions, fixed step to maintain aspect and centre
    
    const float width = float(maxX -  minX + 1);
    const float height = float(maxY - minY + 1);
    
    const float step = std::max(width, height) / 20.0f;
    
    const float centreX = (float(minX) + float(maxX)) / 2.0f;
    const float centreY = (float(minY) + float(maxY)) / 2.0f;
    
    const float weighted_centreX = weighted_x / total_weight;
    const float weighted_centreY = weighted_y / total_weight;
    
    std::cout << "Centre: " << centreX << "," << centreY << " Weighted: " << weighted_centreX << "," << weighted_centreY << std::endl;
 
    //bilinear interpolation
    
    for(std::size_t yDest = 0; yDest < 28; yDest++)
        for(std::size_t xDest = 0; xDest < 28; xDest++)
        {
            const float xSrc = weighted_centreX + (float(xDest) - 13.5f) * step;
            int x1 = std::floor(xSrc);
            int x2 = std::ceil(xSrc);
            float dx = xSrc - x1;
            
            const float ySrc = weighted_centreY + (float(yDest) - 13.5f) * step;
            int y1 = std::floor(ySrc);
            int y2 = std::ceil(ySrc);
            float dy = ySrc - y1;
            
            //leave black if virtual square extends beyond original
            if(xSrc < 0.0f || xSrc > float(sizeofside - 1) || ySrc < 0.0f || ySrc > float(sizeofside - 1)) continue;
            
            output[yDest * 28 + xDest] =    (1-dx)*(1-dy)*input[y1*sizeofside+x1] +
                                            dx*(1-dy)*input[y1*sizeofside+x2] +
                                            (1-dx)*dy*input[y2*sizeofside+x1] +
                                            dx*dy*input[y2*sizeofside+x2];
        }
    
    return output;
    
}
