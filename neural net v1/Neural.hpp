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

using layer = NeuronLayer;

class Neural
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
    void print_stats(std::ostream& ostream);
    
    //Getters + setters
    void set_input(std::size_t node, Scalar value); //directly change a single input
    void set_input(std::vector<float>); // directly change all inputs from a vector
    void set_input(data_set& data,std::size_t index); //directly change all inputs by selecting an entry from a data set
    void set_bias(std::size_t layer, std::vector<Scalar>bias);
    void set_weight(std::size_t layer, std::size_t node, std::vector<Scalar>weight);
    Scalar get_output(std::size_t node);
    std::size_t find_highest_output(void);
    
    //Public methods
    void propagate();
    void gradient_descent(std::size_t trainingSize, double learningRate);
    double trainBatch(const data_set& training_data, const std::span<std::size_t> batch);
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    std::pair<std::size_t, double> predict(const std::vector<float>& input);
    
private:
    // Internal data structure
    std::size_t m_layers = 0;
    std::size_t m_max_layers = 0;
    std::vector<layer> m_layer;

    //Internal methods
    Scalar cost_function(const std::vector<Scalar>& target);
    void zero_training_error();

};






#endif /* neural_hpp */

