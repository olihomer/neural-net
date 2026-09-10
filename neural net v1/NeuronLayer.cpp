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
    std::size_t inputs = weight.cols();
    file.write(reinterpret_cast<const char*>(&inputs),sizeof(inputs));

    //Weights
    file.write(reinterpret_cast<const char*>(weight.data()),weight.size()*sizeof(Scalar));

    //Biases
    file.write(reinterpret_cast<const char*>(bias.data()),bias.size()*sizeof(Scalar));
}

void NeuronLayer::load(std::ifstream& file)
{
    //Size of layer
    file.read(reinterpret_cast<char*>(&size),sizeof(size));
    
    
    //Number of inputs
    std::size_t inputs;
    if(!file.read(reinterpret_cast<char*>(&inputs),sizeof(inputs)))
        throw std::runtime_error("Could not read input layer count");
    
    activation = Matrix(size,1);
    pre_activation = Matrix(size,1);
    weight = Matrix(size,inputs);
    bias = Matrix(size,1);
    error = Matrix(size,1);
    bias_gradient = Matrix(size,1);
    weight_gradient = Matrix(size,inputs);


    //Weights
    file.read(reinterpret_cast<char*>(weight.data()),weight.size()*sizeof(Scalar));

    //Biases
    file.read(reinterpret_cast<char*>(bias.data()),bias.size()*sizeof(Scalar));
}
