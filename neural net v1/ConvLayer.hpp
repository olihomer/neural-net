//
//  ConvLayer.hpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#ifndef ConvLayer_hpp
#define ConvLayer_hpp

#include <cstddef>
#include <stdio.h>
#include "Tensor.hpp"
#include "NeuralTypes.hpp"
#include <vector>
#include <queue>
#include <random>

struct ConvCache
{
    Tensor input;
    Tensor activation;
    Tensor maxPoolSource;
};

class ConvLayer
{
public:
    ConvLayer(std::size_t outputChannels,
              std::size_t inputChannels,
              std::size_t inputHeight,
              std::size_t inputWidth);
    
    void setKernel(std::size_t outputChannel, std::size_t inputChannel, const std::vector<Scalar>& data);
    const Tensor& forward(const Tensor& input);
    Tensor backward(const Tensor& outputGradient, const bool returnInputGradient);
    void print() const;
    void zeroGradients();
    void gradient_descent(std::size_t batchSize, const Scalar learningRate);
    void pushCache();
    void popCache();
    std::size_t getInputWidth() const {return input_.dim(2);};
    std::size_t getInputHeight() const {return input_.dim(1);};
    std::size_t getOutputChannels() const {return pooled_.dim(0);};
    std::size_t getOutputHeight() const {return pooled_.dim(1);};
    std::size_t getOutputWidth() const {return pooled_.dim(2);};
    std::size_t inputChannels() const {return input_.dim(0);};
    std::size_t inputHeight() const {return input_.dim(1);};
    std::size_t inputWidth() const {return input_.dim(2);};
    Scalar inputValue(std::size_t channel, std::size_t y, std::size_t x) const;
    std::size_t activationChannels() const {return activation_.dim(0);};
    std::size_t activationHeight() const {return activation_.dim(1);};
    std::size_t activationWidth() const {return activation_.dim(2);};
    Scalar activationValue(std::size_t channel, std::size_t y, std::size_t x) const;
    std::size_t kernelOutputChannels() const {return kernels_.dim(0);};
    std::size_t kernelInputChannels() const {return kernels_.dim(1);};
    std::size_t kernelHeight() const {return kernels_.dim(2);};
    std::size_t kernelWidth() const {return kernels_.dim(3);};
    Scalar kernelValue(std::size_t outputChannel, std::size_t inputChannel, std::size_t y, std::size_t x) const;
    
    void save(std::ofstream& file) const;
    void load(std::ifstream& file);
    
    static constexpr std::size_t stride = 2;
    static constexpr Scalar beta1 = 0.9f;
    static constexpr Scalar beta2 = 0.999f;
    static constexpr Scalar epsilon = 1e-8f;
    inline static Scalar beta1pow = 1.0f;
    inline static Scalar beta2pow = 1.0f;
    
private:
    Tensor kernels_; //outputs, inputs, kernelY, kernelX
    Tensor kernelGradient_; //outputs, inputs, kernelY, kernelX
    Tensor kernel_m_;
    Tensor kernel_v_;
    Tensor input_;
    Tensor activation_;
    Tensor pooled_;
    Tensor maxPoolSource_;
    static std::mt19937 rng_;
    static std::random_device rd_;
    
    std::vector<Scalar> biases_;
    std::vector<Scalar> biasGradient_;
    std::vector<Scalar> bias_m_;
    std::vector<Scalar> bias_v_;
    
    std::queue<ConvCache> cache_;
    void initialiseWeights();
    
    void convolve_();
    void maxPool_();
    
};

#endif /* ConvLayer_hpp */
