//
//  CNN.cpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#include "CNN.hpp"
#include <iostream>



CNN::CNN(std::size_t inputX, std::size_t inputY, std::size_t inputChannels, std::size_t kernelLayers)
:input_(Tensor({inputChannels, inputX, inputY})), output_(Tensor({kernelLayers, inputX, inputY})), postMaxPool_(Tensor({kernelLayers, inputX/stride, inputY/stride})), maxPoolSource_(Tensor({kernelLayers, inputX/stride, inputY/stride})), kernel_(Tensor({kernelLayers,3,3}))
{
    ;
}

void CNN::setKernel(std::size_t layer, const std::vector<Scalar>& data)
{
    auto d = data.begin();
    for(std::size_t j=0; j<kernel_.dim(2); j++)
        for(std::size_t i=0; i<kernel_.dim(1); i++)
            kernel_(layer,i,j) = *(d++);
}

void CNN::setInput(const std::vector<Scalar>& data)
{
    auto d = data.begin();
    for(std::size_t j=0; j<input_.dim(2); j++)
        for(std::size_t i=0; i<input_.dim(1); i++)
            input_(0,i,j) = *(d++);
}

void CNN::convolve()
{
    auto inputX = input_.dim(1);
    auto inputY = input_.dim(2);
    auto layers = kernel_.dim(0);
    
    //same-padding with integer loops to handle edges more easily
    for(std::size_t layer=0;layer<layers;layer++)
        for(int j=0;j<inputY;j++)
            for(int i=0;i<inputX;i++)
            {
                Scalar sum = 0;
                for(int n=-1;n<2;n++)
                    for(int m=-1;m<2;m++)
                    {
                        sum += (j+n<0 || j+n>inputY-1 || i+m<0 || i+m>inputX-1) ? 0.0f : input_(0,i+m,j+n) * kernel_(layer,m+1,n+1);
                    }
                output_(layer,i,j) = sum > 0.0 ? 1.0 : 0.0;
            }
    
}


void CNN::maxPool()
{
    
    std::size_t inputX = output_.dim(1);
    std::size_t inputY = output_.dim(2);
    std::size_t layers = output_.dim(0);
    std::size_t outputX = inputX/stride;
    std::size_t outputY = inputY/stride;
    
    std::vector<Scalar> candidates;
    candidates.resize(stride*stride);
    
    for(std::size_t layer=0;layer<layers;layer++)
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
                        candidates[x+y*stride]=output_(layer,indexX + x,indexY + y);
                    }
                auto max = std::max_element(candidates.begin(), candidates.end());
                postMaxPool_(layer,i,j) = *max;
                maxPoolSource_(layer,i,j) = max - candidates.begin();
                indexX += stride;
            }
            indexY += stride;
            indexX = 0;
        }
    }
}


void CNN::print() const
{
    kernel_.print();
    
    std::cout << "Kernel data:" << std::endl;
    
    for(std::size_t l=0; l<kernel_.dim(0); l++)
    {
        std::cout << "Layer " << l << std::endl;
        for(std::size_t j=0; j<kernel_.dim(2); j++)
        {
            for(std::size_t i=0; i<kernel_.dim(1); i++)
                std::cout << kernel_(l,i,j) << " ";
            std::cout << std::endl;
        }
    }
    
    std::cout << "Output data:" << std::endl;
    
    for(std::size_t l=0; l<output_.dim(0); l++)
    {
        std::cout << "Layer " << l << std::endl;
        for(std::size_t j=0; j<output_.dim(2); j++)
        {
            for(std::size_t i=0; i<output_.dim(1); i++)
                std::cout << output_(l,i,j) << " ";
            std::cout << std::endl;
        }
    }
    
    std::cout << "MaxPool data:" << std::endl;
    
    for(std::size_t l=0; l<postMaxPool_.dim(0); l++)
    {
        std::cout << "Layer " << l << std::endl;
        for(std::size_t j=0; j<postMaxPool_.dim(2); j++)
        {
            for(std::size_t i=0; i<postMaxPool_.dim(1); i++)
                std::cout << postMaxPool_(l,i,j) << " ";
            std::cout << std::endl;
        }
    }
    
}
