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
#include <string>
#include <vector>

namespace {
Sigmoid hiddenActivationFunction;
Sigmoid outputActivationFunction;
}

AppEngine::AppEngine()
: net({784,128,10}, hiddenActivationFunction, outputActivationFunction)
{
    std::cout << "Constructing Engine" << std::endl;
}

AppEngine::~AppEngine()
{
    std::cout << "Deconstructing Engine" << std::endl;
}

int AppEngine::runApp(void(*progress)(int32_t,double))
{
    mnist_data mnist_training_data("/Users/oliverhomer/Xcode/neural net v1/mnist_test.csv",1000);
    
    //mnist_training_data.print_data(std::cout);
    
    double total_error;
 
    for (int i=0;i<1000;i++)
    {
        total_error = net.train(mnist_training_data);
        net.gradient_descent();
        if(i % 50==0)
        {
            std::cout << i << " ";
            std::cout << "Total error: " << total_error << std::endl;
            net.print_stats(std::cout);
            progress(i,total_error);
        }
    }
    
    
   for(int i=0;i<50;i++)
    {
        int guess = rand()%100;
        int guess_label = mnist_training_data.get_label(guess);
        std::cout << "Guess = " << guess_label;
        
        net.set_input(mnist_training_data, guess);
        net.propagate();
        
        std::cout << ". Net guessed " << net.find_highest_output() << " with value of " << net.get_output(net.find_highest_output()) << std::endl;
        if(net.find_highest_output()!=guess_label)std::cout<<"WRONG!"<<std::endl;
    }
    
    
    return 0;
}

std::pair<int,float> AppEngine::sendRasterData(const float *data, std::size_t size)
{
    if(data==nullptr)return std::pair<int,float>(0,0);

    std::vector<float> vectorData;
    vectorData.resize(size);
    
    for(std::size_t i = 0; i < size; i++)
        vectorData[i] = data[i];
    
    std::cout << "Raster before processing:" << std::endl;
    
    for(std::size_t y = 0; y < 28; y++)
    {
        for(std::size_t x = 0; x < 28; x++)
            std::cout << (vectorData[x+y*28] > 50.0f/255.0f ? "X " : "  ");
        std::cout << std::endl;
    }
    
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
    net.print_values(std::cout);
    std::cout << ". Net guessed " << net.find_highest_output() << " with value of " << net.get_output(net.find_highest_output()) << std::endl;
    
    return std::pair<int,float>(net.find_highest_output(),net.get_output(net.find_highest_output()));
     
}


std::vector<float> AppEngine::preProcess(std::vector<float> input)
{
    std::vector<float> output;
    output.resize(input.size());
    
    std::size_t minX = 27;
    std::size_t maxX = 0;
    std::size_t minY = 27;
    std::size_t maxY = 0;
    
    const float threshold = 5.0f/255.0f;
    
    //establish bounding box
    
    for(std::size_t y = 0; y < 28; y++)
        for(std::size_t x = 0; x < 28; x++)
        {
            if(input[y*28+x] > threshold)
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
    
    //bilinear interpolation
    
    for(std::size_t yDest = 0; yDest < 28; yDest++)
        for(std::size_t xDest = 0; xDest < 28; xDest++)
        {
            const float xSrc = centreX + (float(xDest) - 13.5f) * step;
            int x1 = std::floor(xSrc);
            int x2 = std::ceil(xSrc);
            float dx = xSrc - x1;
            
            const float ySrc = centreY + (float(yDest) - 13.5f) * step;
            int y1 = std::floor(ySrc);
            int y2 = std::ceil(ySrc);
            float dy = ySrc - y1;
            
            //leave black if virtual square extends beyond original
            if(xSrc < 0.0f || xSrc > 27.0f || ySrc < 0.0f || ySrc > 27.0f) continue;
            
            output[yDest * 28 + xDest] =    (1-dx)*(1-dy)*input[y1*28+x1] +
                                            dx*(1-dy)*input[y1*28+x2] +
                                            (1-dx)*dy*input[y2*28+x1] +
                                            dx*dy*input[y2*28+x2];
        }
    
    return output;
    
}
