//
//  ActivationFunction.cpp
//  neural net v1
//
//  Created by Oliver Homer on 12/08/2026.
//

#include "ActivationFunction.hpp"
#include <cmath>

void Sigmoid::activate(const std::vector<double>& z, std::vector<double>& a) const
{
    for(std::size_t i = 0; i < z.size(); i++)
        a[i]=(1/(1+ exp(-z[i])));
}

double Sigmoid::derivative(double preactivation) const
{
    std::vector<double> z = {preactivation};
    std::vector<double> a = {0.0};
    activate(z,a);
    return a[0]*(1-a[0]);
}

void Relu::activate(const std::vector<double>& z, std::vector<double>& a) const
{
for(std::size_t i = 0; i < z.size(); i++)
    a[i] = z[i] > 0.0 ? z[i] : 0.0;
}

double Relu::derivative(double preactivation) const
{
    return preactivation > 0.0 ? 1.0 : 0.0;
}

void Softmax::activate(const std::vector<double>& z, std::vector<double>& a) const
{
   ;
}


double Softmax::derivative(double preactivation) const
{
    throw std::logic_error("Softmax derivative not implemented");
}
