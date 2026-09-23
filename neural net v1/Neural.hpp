//
//  neural.hpp
//  neural net v1
//
//  Created by Oliver Homer on 27/02/2024.
//

#ifndef neural_hpp
#define neural_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <span>
#include "data_set.hpp"
#include "NeuronLayer.hpp"
#include "ActivationFunction.hpp"
#include "NeuralTypes.hpp"
#include "Trainable.hpp"

using layer = NeuronLayer;

class Neural : public Trainable
{
public:
    
    //Constructor
    Neural(std::vector<int> nodes_per_layer, const ActivationType hiddenActivationType, const ActivationType outputActivationType);
    void configure(std::vector<int> nodes_per_layer, const ActivationType hiddenActivationType, const ActivationType outputActivationType);
    
    //Debug
    void print_dimensions(std::ostream& stream);
    void print_weights(std::ostream& stream);
    void print_biases(std::ostream& stream);
    void print_values(std::ostream& stream);
    void print_errors(std::ostream& stream);
    void print_training_errors(std::ostream& stream);
    void print_network(std::ostream& stream);
    void print_stats(std::ostream& ostream) override;
    
    //Getters + setters
    std::size_t nInputs_(){return layer_[0].size;};
    std::size_t nOutputs_(){return layer_[layers_-1].size;};
    
    void set_input(std::size_t node, Scalar value); //directly change a single input
    void set_input(const std::vector<Scalar>&); // directly change all inputs from a vector
    void set_input(data_set& data,std::size_t index); //directly change all inputs by selecting an entry from a data set
    std::size_t numLayers(){return layers_;};
    Matrix get_input_error(){return layer_[0].error;};

    Scalar get_output(std::size_t node);
    std::vector<Scalar> get_output();
    std::size_t find_highest_output(void);
    
    //Public methods
    void propagate();
    void propagateBatch();
    void gradient_descent(std::size_t trainingSize, double learningRate) override;
    double trainBatch(const data_set& training_data, const std::span<const std::size_t> batch) override;
    double trainBatch(const Matrix& inputs, const Matrix& targets);

    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void save(std::ofstream& stream) const;
    void load(std::ifstream& stream);

    std::pair<std::size_t, Scalar> predict(const std::vector<Scalar>& input);
    
private:
    // Internal data structure
    std::size_t layers_ = 0;
    std::size_t max_layers_ = 0;
    std::vector<layer> layer_;

    //Internal methods
    Scalar cost_function(const std::vector<Scalar>& target);
    void zero_training_error();

};






#endif /* neural_hpp */

