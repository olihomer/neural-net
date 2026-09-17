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
    Tensor backward(const Tensor& outputGradient, bool returnInputGradient);
    void print() const;
    void zeroGradients();
    void gradient_descent(const Scalar scale);
    void pushCache();
    void popCache();

    
    static constexpr std::size_t stride = 2;
    
private:
    Tensor kernels_; //outputs, inputs, kernelY, kernelX
    Tensor kernelGradient_; //outputs, inputs, kernelY, kernelX
    Tensor input_;
    Tensor activation_;
    Tensor pooled_;
    Tensor maxPoolSource_;
    
    std::vector<Scalar> biases_;
    std::vector<Scalar> biasGradient_;
    std::queue<ConvCache> cache_;
    
    void convolve_();
    void maxPool_();
    
};

#endif /* ConvLayer_hpp */
