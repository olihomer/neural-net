//
//  CNN.hpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#ifndef CNN_hpp
#define CNN_hpp

#include <cstddef>
#include <stdio.h>
#include "Tensor.hpp"
#include "NeuralTypes.hpp"
#include <vector>

class CNN
{
public:
    CNN(std::size_t inputX, std::size_t inputY, std::size_t inputChannels, std::size_t kernelLayers); //fixed 3x3 kernel
    void setKernel(std::size_t layer, const std::vector<Scalar>& data);
    void setInput(const std::vector<Scalar>& data);
    void convolve();
    void maxPool();
    void print() const;
    const std::size_t stride = 2;
    std::vector<Scalar> flatten(){return std::vector<Scalar>(postMaxPool_.data(),postMaxPool_.data() + postMaxPool_.size());};
    
private:
    Tensor kernel_;
    Tensor input_;
    Tensor output_;
    Tensor postMaxPool_;
    Tensor maxPoolSource_;
};





#endif /* CNN_hpp */
