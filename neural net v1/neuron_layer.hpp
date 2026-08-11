#ifndef neuron_layer_hpp
#define neuron_layer_hpp

#include <vector>
#include <cstddef>

// A simple, explicit container for a neural-network layer's state.
// This mirrors the existing 'layer' struct fields to enable a staged migration
// without obscuring the forward/backward equations.
class NeuronLayer {
public:
    std::size_t size{};
    std::vector<double> pre_activation;
    std::vector<double> error;
    std::vector<double> bias_gradient;

    std::vector<double> activation;
    std::vector<double> bias;
    std::vector<std::vector<double>> weight;
    std::vector<std::vector<double>> weight_gradient;

    NeuronLayer() = default;

    explicit NeuronLayer(std::size_t size_)
        : size(size_)
    {
        activation.resize(size);
        pre_activation.resize(size);
        error.resize(size);
        bias_gradient.resize(size);
        bias.resize(size);
        weight.resize(size);
        weight_gradient.resize(size);
    }

    // Ensure incoming connection dimensions match the previous layer.
    void resize_for_previous(std::size_t prev_size) {
        for (std::size_t j = 0; j < size; ++j) {
            weight[j].resize(prev_size);
            weight_gradient[j].resize(prev_size);
        }
    }

    // Zero accumulated gradients for this layer given the previous layer's size.
    void zero_gradients(std::size_t prev_size) {
        for (std::size_t j = 0; j < size; ++j) {
            bias_gradient[j] = 0.0;
            for (std::size_t k = 0; k < prev_size; ++k) {
                weight_gradient[j][k] = 0.0;
            }
        }
    }
};

#endif // neuron_layer_hpp
