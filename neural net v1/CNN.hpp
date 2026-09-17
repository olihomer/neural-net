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
    
    std::vector<Scalar> forward(const Tensor& input);
    Tensor backward(const Tensor& outputGradient);

    void print() const;
    ConvLayer conv1_;
    ConvLayer conv2_;
    Neural classifier_;
    
    void gradient_descent(std::size_t trainingSize, double learningRate) override;
    double trainBatch(const data_set& training_data, const std::span<const std::size_t> batch) override;
    void print_stats(std::ostream& ostream) override;
    
    void set_inputMLP(std::vector<Scalar>& input){classifier_.set_input(input);};
    void propagateMLP(){classifier_.propagate();};
    Scalar get_output(std::size_t node){return classifier_.get_output(node);};
    std::size_t find_highest_output(void){return classifier_.find_highest_output();};
    
    
private:
    
    std::vector<Scalar> flatten_(const Tensor& input){return std::vector<Scalar>(input.data(),input.data() + input.size());};
};


#endif /* CNN_hpp */
