//
//  EmnistData.cpp
//  neural net v1
//
//  Created by Oliver Homer on 06/10/2026.
//

#include "EmnistData.hpp"
#include <iostream>
#include <cstddef>
#include <bit>
#include <algorithm>

EmnistData::EmnistData(const std::string& fileprefix, std::size_t size)
{
    //Emnist data file loader
    std::string imageFilename = fileprefix + "-images-idx3-ubyte";
    std::string labelFilename = fileprefix + "-labels-idx1-ubyte";
    
    std::ifstream imageFile(imageFilename, std::ios::binary);
    std::ifstream labelFile(labelFilename, std::ios::binary);
    
    if(!imageFile)
        throw std::runtime_error("EMNIST image file could not be opened");
    
    if(!labelFile)
        throw std::runtime_error("EMNIST label file could not be opened");
    
    //read image file
    
    std::cout << "Image and label files opened ok. Processing image file." << std::endl;
    
    //magic number, get dimensions
    std::byte magicNumber[4];
    
    for(std::size_t i = 0; i < 4; i++)
        imageFile.read(reinterpret_cast<char*>(&magicNumber[i]),sizeof(magicNumber[i]));
    
    std::size_t dataType = (int)magicNumber[2];
    if(dataType!=8)throw std::runtime_error("Unsupported datatype.");
    
    std::size_t dimensions = (int)magicNumber[3];
    
    //get size of each dimension
    
    std::vector<size_t> dimSize;
    
    for(std::size_t i = 0; i < dimensions; i++)
    {
        std::int32_t inputInt32;
        imageFile.read(reinterpret_cast<char*>(&inputInt32),sizeof(inputInt32));
        dimSize.emplace_back(__builtin_bswap32(inputInt32));
    }
    
    size_ = std::min(size,dimSize[0]);
    inputX_ = dimSize[2];
    inputY_ = dimSize[1];
    n_inputs_ = inputX_ * inputY_;
    n_outputs_ = labelSet.length();
    
    //read images
    
    std::cout << "Reading first " << size_ << " images of " << dimSize[0] << " in file." << std::endl;
    std::cout << "Image dimensions = " << inputX_ << "x" << inputY_ << " = " << n_inputs_ << " inputs." << std::endl;
    
    data_.resize(size_);
    label_.resize(size_);
    std::vector<Scalar> temp(n_inputs_);
    
    for(std::size_t n = 0; n < size_ ; n++)
    {
        data_[n].inputs.resize(n_inputs_);
        data_[n].outputs.resize(n_outputs_);
     
        for(std::size_t i = 0; i < n_inputs_ ; i++)
        {
            //read into temp before reorienting
            std::uint8_t inputInt8;
            imageFile.read(reinterpret_cast<char*>(&inputInt8),sizeof(inputInt8));
            temp[i] = (float)inputInt8/255.0f;
        }
        
        //reorient images, store in class
        for(std::size_t y = 0; y < inputY_ ; y++)
            for(std::size_t x = 0; x < inputX_; x++)
                data_[n].inputs[y*inputX_+x] = temp[x*inputY_+y];
    }
    
    //read label file
    
    std::cout << "Processing label file." << std::endl;

    //magic number, get dimensions
    
    for(std::size_t i = 0; i < 4; i++)
    {
        labelFile.read(reinterpret_cast<char*>(&magicNumber[i]),sizeof(magicNumber[i]));
    }
    
    dataType = (int)magicNumber[2];
    if(dataType!=8)throw std::runtime_error("Unsupported datatype.");
    
    dimensions = (int)magicNumber[3];
    
    if((int)dimensions!=1)throw std::runtime_error("Label file contains multiple dimensions.");
    
    //check number of labels
    
    std::int32_t inputInt32;
    labelFile.read(reinterpret_cast<char*>(&inputInt32),sizeof(inputInt32));
    std::size_t labelLength = __builtin_bswap32(inputInt32);
    
    if(labelLength!=dimSize[0])throw("Label file length does not match image file.");
    
    //read labels
    
    std::cout << "Reading first " << size_ << " labels of " << labelLength << " in file." << std::endl;

    //labels should range from 0 to 46
    for(std::size_t n = 0; n < size_; n++)
    {
        std::uint8_t inputInt8;
        labelFile.read(reinterpret_cast<char*>(&inputInt8),sizeof(inputInt8));
        label_[n] = (int)inputInt8;
        //store output values (0 or 1)
        for(std::size_t i = 0; i < n_outputs_; i++)
            data_[n].outputs[i] = (label_[n] == i) ? 1 : 0;
    }

    std::cout << "Files read successfully." << std::endl;
    imageFile.close();
    labelFile.close();

    //test print
    
    /*for(std::size_t n = 0; n < size_ ; n++)
    {
        std::cout << "Label: " << labelSet[label_[n]];
        for(std::size_t y = 0; y < inputY_ ; y++)
        {
            std::cout << std::endl;
            for(std::size_t x = 0; x < inputX_; x++)
                std::cout << (data_[n].inputs[y*inputX_+x]>0.1f ? "X":" ") << " ";
        }
    }*/
}


const int EmnistData::get_label(int index)
{
    return label_[index];
}
