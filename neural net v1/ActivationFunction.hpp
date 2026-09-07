//
//  ActivationFunction.hpp
//  neural net v1
//
//  Created by Oliver Homer on 11/08/2026.
//

#include <vector>

#ifndef ActivationFunction_hpp
#define ActivationFunction_hpp

class ActivationFunction
{
public:
    virtual void activate(const std::vector<double>& z, std::vector<double>& a) const = 0;
    virtual double derivative(double preactivation) const = 0;
    virtual ~ActivationFunction() = default;
};

class Sigmoid final : public ActivationFunction
{
public:
    void activate(const std::vector<double>& z, std::vector<double>& a) const override;
    double derivative(double preactivation) const override;
};

class Relu final : public ActivationFunction
{
public:
    void activate(const std::vector<double>& z, std::vector<double>& a) const override;
    double derivative(double preactivation) const override;
};

class Softmax final : public ActivationFunction
{
public:
    void activate(const std::vector<double>& z, std::vector<double>& a) const override;
    double derivative(double preactivation) const override;
};

#endif // !ActivationFunction_hpp

