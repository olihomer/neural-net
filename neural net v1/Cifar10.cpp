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
    
    std::cout << "CIFAR10 file opened ok. Processing images." << std::endl;
    
    size_ = std::min((int)size, 10000);
    
    data_.resize(size);
    n_inputs_ = 3072; // 3 x 1024 x 1024
    n_outputs_ = 10;
    
    std::uint8_t inputByte;
    
    for(std::size_t i = 0; i < size_; i++)
    {
        data_[i].inputs.resize(n_inputs_);
        data_[i].outputs.resize(n_outputs_);
        
        imageFile.read(reinterpret_cast<char*>(&inputByte),sizeof(inputByte));
        label_[i] = inputByte;
        
        for(std::size_t j = 0; j < n_inputs_; j++)
        {
            imageFile.read(reinterpret_cast<char*>(&inputByte),sizeof(inputByte));
            data_[i].inputs[j] = inputByte;
            std::cout << inputByte << std::endl;
        }
    }
}
