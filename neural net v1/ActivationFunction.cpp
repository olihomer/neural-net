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

Scalar Sigmoid::activateScalar(Scalar z) const
{
        return (1/(1+ exp(-z)));
}

Matrix Sigmoid::activate(const Matrix& input) const
{
    return ElementalActivationFunction::activate(input);
}

Matrix Sigmoid::derivative(const Matrix& input) const
{
    return ElementalActivationFunction::derivative(input);
}

Scalar Sigmoid::derivativeScalar(Scalar a) const
{
    return a * (1.0f - a);
}

Scalar Relu::activateScalar(Scalar z) const
{
    return z > 0.0 ? z : 0.0;
}

Matrix Relu::activate(const Matrix& input) const
{
    return ElementalActivationFunction::activate(input);
}

Matrix Relu::derivative(const Matrix& input) const
{
    return ElementalActivationFunction::derivative(input);
}

Scalar Relu::derivativeScalar(Scalar a) const
{
    return a > 0.0 ? 1.0 : 0.0;
}

Matrix Softmax::activate(const Matrix& z) const
{
    Scalar zmax = z.max();
    Scalar sum = 0;
    Matrix a(z.rows(),z.cols());
    
    for(std::size_t j = 0; j < z.cols(); j++)
    {
        sum = 0;
        zmax = z.colMax(j);
        for(std::size_t i = 0; i < z.rows(); i++)
        {
            a(i,j) = std::exp(z(i,j) - zmax);
            sum += a(i,j);
        }
        for(std::size_t i = 0; i < z.rows(); i++)
            a(i,j) = a(i,j) * (1/sum);
    }
    return a;
}

Matrix Softmax::derivative(const Matrix &a) const
{
    throw std::logic_error("Softmax derivative not implemented");
}
