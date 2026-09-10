//
//  ActivationFunction.hpp
//  neural net v1
//
//  Created by Oliver Homer on 11/08/2026.
//

#include <vector>
#include <cstdint>
#include "NeuralTypes.hpp"
#include "Matrix.hpp"

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
    virtual Matrix activate(const Matrix&) const = 0;
    virtual Matrix derivative(const Matrix& ) const = 0;
    virtual ~ActivationFunction() = default;
    virtual ActivationType type() const noexcept = 0;
};

class ElementalActivationFunction : public ActivationFunction
{
public:
    Matrix activate(const Matrix& input) const override
    {
        return input.apply(
                           [this](Scalar x)
                           {
                               return activateScalar(x);
                           }
                           );
    }
    Matrix derivative(const Matrix& input) const override
    {
        return input.apply(
                           [this](Scalar x)
                           {
                               return derivativeScalar(x);
                           }
                           );
    }
protected:
    virtual Scalar activateScalar(Scalar x) const = 0;
    virtual Scalar derivativeScalar(Scalar x) const = 0;
};

const ActivationFunction& activationFromType(ActivationType type);

class Sigmoid final : public ElementalActivationFunction
{
public:
    Matrix activate(const Matrix& input) const override;
    Matrix derivative(const Matrix& activation) const override;
    Scalar activateScalar(Scalar x) const override;
    Scalar derivativeScalar(Scalar x) const override;
    ActivationType type() const noexcept override {return ActivationType::Sigmoid;};
};

class Relu final : public ElementalActivationFunction
{
public:
    Matrix activate(const Matrix& input) const override;
    Matrix derivative(const Matrix& activation) const override;
    Scalar activateScalar(Scalar x) const override;
    Scalar derivativeScalar(Scalar x) const override;
    ActivationType type() const noexcept override {return ActivationType::Relu;};
};

class Softmax final : public ActivationFunction
{
public:
    Matrix activate(const Matrix& input) const override;
    Matrix derivative(const Matrix& activation) const override;
    ActivationType type() const noexcept override {return ActivationType::Softmax;};
};

#endif // !ActivationFunction_hpp

