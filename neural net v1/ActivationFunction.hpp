//
//  ActivationFunction.hpp
//  neural net v1
//
//  Created by Oliver Homer on 11/08/2026.
//

#include <vector>
#include "NeuralTypes.hpp"

#ifndef ActivationFunction_hpp
#define ActivationFunction_hpp

enum class ActivationType: std::uint8_t
{
    Sigmoid = 1,
    Relu = 2,
    Softmax = 3,
};

class ActivationFunction
{
public:
    virtual void activate(const std::vector<Scalar>& z, std::vector<Scalar>& a) const = 0;
    virtual Scalar derivative(Scalar preactivation) const = 0;
    virtual ~ActivationFunction() = default;
    virtual ActivationType type() const noexcept = 0;
};

const ActivationFunction& activationFromType(ActivationType type);

class Sigmoid final : public ActivationFunction
{
public:
    void activate(const std::vector<Scalar>& z, std::vector<Scalar>& a) const override;
    Scalar derivative(Scalar preactivation) const override;
    ActivationType type() const noexcept override {return ActivationType::Sigmoid;};
};

class Relu final : public ActivationFunction
{
public:
    void activate(const std::vector<Scalar>& z, std::vector<Scalar>& a) const override;
    Scalar derivative(Scalar preactivation) const override;
    ActivationType type() const noexcept override {return ActivationType::Relu;};

};

class Softmax final : public ActivationFunction
{
public:
    void activate(const std::vector<Scalar>& z, std::vector<Scalar>& a) const override;
    Scalar derivative(Scalar preactivation) const override;
    ActivationType type() const noexcept override {return ActivationType::Softmax;};

};

#endif // !ActivationFunction_hpp

