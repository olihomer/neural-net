//
//  CNN.cpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#include "CNN.hpp"
#include <iostream>



CNN::CNN()
:conv1_(ConvLayer(8,1,28,28)), conv2_(ConvLayer(16,8,14,14))
{
    ;
}

void CNN::print() const
{
    ;
    
}


std::vector<Scalar> CNN::forward(const Tensor& input)
{
    auto& x1 = conv1_.forward(input);
    auto& x2 = conv2_.forward(x1);
    
    auto flat = flatten_(x2);
    
    return flat;
}
