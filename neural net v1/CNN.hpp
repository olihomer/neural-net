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
#include "ConvLayer.hpp"

class CNN
{
public:
    CNN();
    
    std::vector<Scalar> forward(const Tensor& input);
    void print() const;
    ConvLayer conv1_;
    ConvLayer conv2_;
private:
    
    std::vector<Scalar> flatten_(const Tensor& input){return std::vector<Scalar>(input.data(),input.data() + input.size());};
};


#endif /* CNN_hpp */
