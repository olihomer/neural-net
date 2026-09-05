//
//  ActivationFunction.cpp
//  neural net v1
//
//  Created by Oliver Homer on 12/08/2026.
//

#include "ActivationFunction.hpp"
#include <cmath>

double Sigmoid::activate(double preactivation) const
{
    return (1/(1+ exp(-preactivation)));
}

double Sigmoid::derivative(double preactivation) const
{
    double z = activate(preactivation);
    return z*(1-z);
}

double Relu::activate(double preactivation) const
{
    return preactivation > 0.0 ? preactivation : 0.0;
}

double Relu::derivative(double preactivation) const
{
    return preactivation > 0.0 ? 1.0 : 0.0;
}
