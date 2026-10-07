//
//  CNN.hpp
//  neural net v1
//
//  Created by Oliver Homer on 14/09/2026.
//

#ifndef CNN_hpp
#define CNN_hpp

#include <cstddef>
#include <stdio.h>
#include "Tensor.hpp"
#include "NeuralTypes.hpp"
#include <vector>
#include "ConvLayer.hpp"
#include "Neural.hpp"
#include "Trainable.hpp"


class CNN : public Trainable
{
public:
    CNN();
    void configure(std::vector<std::size_t> convLayerOutputChannels,
                   std::size_t classifierHiddenLayerSize,
                   Scalar dropout,
                   std::size_t mlpOutputs,
                   std::size_t inputX,
                   std::size_t inputY,
                   std::size_t inputChannels);
    
    const Tensor& forward(const Tensor& input);
    const Tensor& forwardBatch(const Tensor& input);

    void backward(const Tensor& outputGradient);
    void backwardBatch(const GradientView& outputGradient, std::size_t miniBatchIndex);

    
    void print() const;
    std::vector<ConvLayer> convLayer_;
    Neural classifier_;
    
    void gradient_descent(std::size_t trainingSize, double learningRate) override;
    double trainBatch(const DataSet& training_data, const std::span<const std::size_t> batch) override;
    void print_stats(std::ostream& ostream) override;
    
    void set_inputMLP(std::vector<Scalar>& input){classifier_.set_input(input);};
    void propagateMLP(){classifier_.propagate();};
    Scalar get_output(std::size_t node){return classifier_.get_output(node);};
    std::size_t find_highest_output(void){return classifier_.find_highest_output();};
    std::pair<std::size_t, Scalar> predict(const std::vector<Scalar>& input);

    void resetBeta() override {ConvLayer::beta1pow=1.0f;ConvLayer::beta2pow=1.0f;classifier_.resetBeta();}
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    
private:
    std::size_t nLayers_;
};


#endif /* CNN_hpp */
