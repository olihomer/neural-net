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
#include "data_set.hpp"
#include "neuron_layer.hpp"

using layer = NeuronLayer;

class Neural
{
public:
    
    //Constructor
    Neural(std::vector<int> nodes_per_layer, const ActivationFunction& hiddenActivationFunction, const ActivationFunction& outputActivationFunction);
    
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
    void set_input(std::size_t node, double value); //directly change a single input
    void set_input(std::vector<float>); // directly change all inputs from a vector
    void set_input(data_set& data,std::size_t index); //directly change all inputs by selecting an entry from a data set
    void set_bias(std::size_t layer, std::vector<double>bias);
    void set_weight(std::size_t layer, std::size_t node, std::vector<double>weight);
    double get_output(std::size_t node);
    std::size_t find_highest_output(void);

    //Statics
    static double sigmoid(double x){return (1/(1+exp(-x)));}
    static double sigmoid_prime(double x){double z = sigmoid(x);return z*(1-z);}
    const double learning_rate = 0.5;
    
    //Public methods
    void propagate();
    void gradient_descent(const int trainingSize);
    double train(const data_set& training_data);
    
private:
    // Internal data structure
    std::size_t m_layers = 0;
    std::size_t m_max_layers = 0;
    std::vector<layer> m_layer;

    //Internal methods
    double cost_function(const std::vector<double>& target);
    void zero_training_error();

};






#endif /* neural_hpp */

