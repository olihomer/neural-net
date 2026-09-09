//
//  ActivationFunction.cpp
//  neural net v1
//
//  Created by Oliver Homer on 12/08/2026.
//

#include "ActivationFunction.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

const ActivationFunction& activationFromType(ActivationType type)
{
    static const Sigmoid sigmoid;
    static const Relu relu;
    static const Softmax softmax;
    
    switch(type)
    {
        case ActivationType::Sigmoid: return sigmoid;
        case ActivationType::Relu: return relu;
        case ActivationType::Softmax: return softmax;
    }
    throw std::runtime_error("Unknown activation function");
}

void Sigmoid::activate(const std::vector<Scalar>& z, std::vector<Scalar>& a) const
{
    for(std::size_t i = 0; i < z.size(); i++)
        a[i]=(1/(1+ exp(-z[i])));
}

Scalar Sigmoid::derivative(Scalar preactivation) const
{
    std::vector<Scalar> z = {preactivation};
    std::vector<Scalar> a = {0.0};
    activate(z,a);
    return a[0]*(1-a[0]);
}

void Relu::activate(const std::vector<Scalar>& z, std::vector<Scalar>& a) const
{
    for(std::size_t i = 0; i < z.size(); i++)
    a[i] = z[i] > 0.0 ? z[i] : 0.0;
}

Scalar Relu::derivative(Scalar preactivation) const
{
    return preactivation > 0.0 ? 1.0 : 0.0;
}

void Softmax::activate(const std::vector<Scalar>& z, std::vector<Scalar>& a) const
{
    Scalar zmax = *std::max_element(z.begin(), z.end());
    Scalar sum = 0;
    
    for(std::size_t i = 0; i < z.size(); i++)
    {
        a[i] = std::exp(z[i] - zmax);
        sum += a[i];
    }
    
    for(Scalar& value : a)
        value /= sum;
}


Scalar Softmax::derivative(Scalar preactivation) const
{
    throw std::logic_error("Softmax derivative not implemented");
}
