//
//  Cifar10.cpp
//  neural net v1
//
//  Created by Oliver Homer on 07/10/2026.
//

#include "Cifar10.hpp"


#include "EmnistData.hpp"
#include <iostream>
#include <cstddef>
#include <bit>
#include <algorithm>


Cifar10::Cifar10(const std::string& filename, std::size_t size)
{
    //Cifar data file loader
    
    std::ifstream imageFile(filename, std::ios::binary);
    
    if(!imageFile)
        throw std::runtime_error("CIFAR image file could not be opened");
    
    //read image file
    
    std::cout << "CIFAR10 file opened ok." << std::endl;
    
    size_ = std::min((int)size, 10000);
    
    data_.resize(size);
    label_.resize(size);
    n_inputs_ = 3072; // 3 x 32 x 32
    inputX_ = 32;
    inputY_ = 32;
    n_outputs_ = 10;
    
    std::uint8_t inputByte;
    
    std::cout << "Loading " << size_ << " images." << std::endl;
    
    for(std::size_t i = 0; i < size_; i++)
    {
        data_[i].inputs.resize(n_inputs_);
        data_[i].outputs.resize(n_outputs_);
        
        //read label
        imageFile.read(reinterpret_cast<char*>(&inputByte),sizeof(inputByte));
        label_[i] = (int)inputByte;
        std::cout << label_[i] << std::endl;
        
        //store pixels
        for(std::size_t j = 0; j < n_inputs_; j++)
        {
            imageFile.read(reinterpret_cast<char*>(&inputByte),sizeof(inputByte));
            data_[i].inputs[j] = (float)inputByte/255.0f;
        }
        
        //store outputs based on label
        for(std::size_t j = 0; j < n_outputs_; j++)
            data_[i].outputs[j] = (label_[i] == j) ? 1 : 0;
    }
    
    //test print
    for(std::size_t i = 0; i < size_; i++)
    {
        std::cout << Cifar10::labelSet[label_[i]] << std::endl;
        for(std::size_t y = 0; y < inputY_; y++)
        {
            for(std::size_t x = 0; x < inputX_; x++)
            {
                std::cout << (data_[i].inputs[y*inputX_+x] > 0.5f ? "X " : "  ");
            }
            std::cout << std::endl;
        }
    }
}

const int Cifar10::get_label(int index)
{
    return label_[index];
}
