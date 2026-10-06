//
//  EmnistData.cpp
//  neural net v1
//
//  Created by Oliver Homer on 06/10/2026.
//

#include "EmnistData.hpp"


EmnistData::EmnistData(const std::string& fileprefix, std::size_t size)
{
    std::string imageFilename = fileprefix + "-idx3-ubyte";
    std::string labelFilename = fileprefix + "-id1-ubyte";


    std::ifstream imageFile(imageFilename, std::ios::binary);
    std::ifstream labelFile(labelFilename, std::ios::binary);

    
    if(!imageFile)
        throw std::runtime_error("EMNIST image file could not be opened");
    
    if(!labelFile)
        throw std::runtime_error("EMNIST label file could not be opened");
    
    
}
