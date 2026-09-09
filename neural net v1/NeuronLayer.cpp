//
//  neuron_layer.cpp
//  neural net v1
//
//  Created by Oliver Homer on 08/09/2026.
//

#include <fstream>
#include "NeuronLayer.hpp"

void NeuronLayer::save(std::ofstream& file) const
{
    //Size of layer
    file.write(reinterpret_cast<const char*>(&size),sizeof(size));

    //Number of inputs
    std::size_t inputs = weight[0].size();
    file.write(reinterpret_cast<const char*>(&inputs),sizeof(inputs));

    //Weights
    for(std::size_t i = 0; i < size; i++)
        for(std::size_t j = 0; j < weight[i].size(); j++)
            file.write(reinterpret_cast<const char*>(&weight[i][j]),sizeof(weight[i][j]));

    //Biases
    for(std::size_t i = 0; i < size; i++)
        file.write(reinterpret_cast<const char*>(&bias[i]),sizeof(bias[i]));
}

void NeuronLayer::load(std::ifstream& file)
{
    //Size of layer
    file.read(reinterpret_cast<char*>(&size),sizeof(size));
    
    activation.resize(size);
    pre_activation.resize(size);
    error.resize(size);
    bias_gradient.resize(size);

    bias.resize(size);
    weight.resize(size);
    weight_gradient.resize(size);
    
    //Number of inputs
    std::size_t inputs;
    if(!file.read(reinterpret_cast<char*>(&inputs),sizeof(inputs)))
        throw std::runtime_error("Could not read input layer count");
    
    resize_for_previous(inputs);
    
    //Weights
    for(std::size_t i = 0; i < size; i++)
        for(std::size_t j = 0; j < weight[i].size(); j++)
            file.read(reinterpret_cast<char*>(&weight[i][j]),sizeof(weight[i][j]));

    //Biases
    for(std::size_t i = 0; i < size; i++)
        file.read(reinterpret_cast<char*>(&bias[i]),sizeof(bias[i]));
}
