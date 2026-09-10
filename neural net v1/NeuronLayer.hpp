#ifndef neuron_layer_hpp
#define neuron_layer_hpp

#include <vector>
#include <cstddef>
#include "ActivationFunction.hpp"
#include "NeuralTypes.hpp"
#include "Matrix.hpp"
#include <iosfwd>

// A simple, explicit container for a neural-network layer's state.
// This mirrors the existing 'layer' struct fields to enable a staged migration
// without obscuring the forward/backward equations.

class NeuronLayer {
public:
    std::size_t size{};
    const ActivationFunction& activation_function_;
    
    Matrix error;
    Matrix bias_gradient;
    Matrix pre_activation;
    Matrix activation;
    Matrix bias;
    Matrix weight;
    Matrix weight_gradient;
    
    inline static Sigmoid default_activation_function_{};
    
    explicit NeuronLayer (const ActivationFunction& af = default_activation_function_)
        : activation_function_(af), weight(0,0), activation(0,0), bias(0,0), pre_activation(0,0), error(0,0), bias_gradient(0,0), weight_gradient(0,0)
    {
        ;
    }

    void save(std::ofstream& file) const;
    
    void load(std::ifstream& file);
    

    void zero_gradients(std::size_t prev_size)
    {
        bias_gradient = Matrix(bias_gradient.rows(),bias_gradient.cols());
        weight_gradient = Matrix(weight_gradient.rows(),weight_gradient.cols());
    }
};

#endif // neuron_layer_hpp
