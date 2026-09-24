//
//  neural.cpp
//  neural net v1
//
//  Created by Oliver Homer on 27/02/2024.
//

#include "Neural.hpp"
#include "NeuronLayer.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include "Matrix.hpp"
#include <vector>



std::pair<std::size_t, Scalar> Neural::predict(const std::vector<Scalar>& input)
{
    //resize pre_activation, activation and error matrices to hold column vectors
    for(auto &layer: layer_)
    {
        layer.pre_activation = Matrix(layer.size,1);
        layer.activation = Matrix(layer.size,1);
        layer.error = Matrix(layer.size,1);
    }
    
    set_input(input);
    propagate();
    const auto index = find_highest_output();
    return {index,get_output(index)};
}

void Neural::load(std::ifstream& file)
{
    //Magic number
    char MAGIC[] = {'X','X','X','X'};
    file.read(reinterpret_cast<char*>(&MAGIC), sizeof(MAGIC));
    
    if(std::string(MAGIC,sizeof(MAGIC))!="NNET")
        throw std::runtime_error("Not a valid neural network file");
    
    //Version
    std::uint8_t version;
    file.read(reinterpret_cast<char*>(&version),sizeof(version));
    
    if(version!=3)
        throw std::runtime_error("Unsupported file version");
    
    //Number of layers
    file.read(reinterpret_cast<char*>(&layers_),sizeof(layers_));
    
    layer_.clear();

    //Activation functions, create layers
    for(std::size_t i = 0 ; i < layers_; i++)
    {
        std::uint8_t activation;
        file.read(reinterpret_cast<char*>(&activation),sizeof(activation));
        layer_.emplace_back(activationFromType(static_cast<ActivationType>(activation)));
        
        
    }
    
    max_layers_ = 0;
    
    //Load each layer
    for(auto &layer: layer_)
    {
        layer.load(file);
        if(layer.size > max_layers_)max_layers_ = layer.size;
    }
    
    if(!file)
        throw std::runtime_error("Failed while loading");
}


void Neural::load(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    
    if(!file)
        throw std::runtime_error("Load file could not be opened");
    
    load(file);
    }


void Neural::save(std::ofstream& file) const
{
    //Magic number
    constexpr char MAGIC[] = {'N','N','E','T'};
    file.write(MAGIC, sizeof(MAGIC));
    
    //Version
    const uint8_t version = 3;
    file.write(reinterpret_cast<const char*>(&version),sizeof(version));
    
    //Number of layers
    file.write(reinterpret_cast<const char*>(&layers_),sizeof(layers_));
    
    //Activation functions
    for(std::size_t i = 0 ; i < layers_; i++)
    {
        const std::uint8_t activation = (std::uint8_t)layer_[i].activation_function_.type();
        file.write(reinterpret_cast<const char*>(&activation),sizeof(activation));
    }
    
    //Save each layer
    for(const auto &layer: layer_)
    {
        layer.save(file);
    }
    
    if(!file)
        throw std::runtime_error("Failed while saving");
}
        
void Neural::save(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);
    
    if(!file)
        throw std::runtime_error("Save file could not be opened");
    
    save(file);
}

double Neural::trainBatch(const data_set &training_data, const std::span<const std::size_t> batch)
{
    if(training_data.n_inputs() != layer_[0].size || training_data.n_outputs() != layer_[layers_-1].size)
    {
        std::cout << "Error: training data size does not match net topology" << std::endl;
        std::cout << "Inputs: " << training_data.n_inputs() << " in data versus " << layer_[0].size << " in net." << std::endl;
        std::cout << "Outputs: " << training_data.n_outputs() << " in data versus " << layer_[layers_-1].size << " in net." << std::endl;
        exit(1);
    }
    
    Matrix inputs(layer_[0].size,batch.size());
    Matrix targets(training_data.n_outputs(),batch.size());
    
    for(std::size_t col = 0; col < batch.size(); col++)
    {
        for(std::size_t row = 0; row < training_data.n_inputs(); row++)
            inputs(row,col) = training_data.get_data()[batch[col]].inputs[row];
        for(std::size_t row = 0; row < training_data.n_outputs(); row++)
            targets(row,col) = training_data.get_data()[batch[col]].outputs[row];
    }
    
    return trainBatch(inputs, targets);
}



double Neural::trainBatch(const Matrix& inputs, const Matrix& targets)
{
    if(inputs.rows() != layer_[0].size ||targets.rows() != layer_[layers_-1].size)
    {
        std::cout << "Error: training data size does not match net topology" << std::endl;
        std::cout << "Inputs: " << inputs.rows() << " in matrix versus " << layer_[0].size << " in net." << std::endl;
        std::cout << "Outputs: " << targets.rows() << " in matrix versus " << layer_[layers_-1].size << " in net." << std::endl;
        exit(1);
    }
    
    zero_training_error();
    
    auto& input = layer_[0];
    auto& output = layer_[layers_-1];
    
    //set input layer activations and target output activations for entire batch

    input.activation = inputs;

    //propagate through network
    
    propagateBatch(true);
        
    const auto& probabilities = output.activation;
    double total_loss = 0.0;
    
    for(std::size_t col = 0; col < targets.cols(); col++)
    {
        for(std::size_t row = 0; row < targets.rows(); row++)
        {
            if(targets(row,col) == 1.0f)
            {
                const Scalar p = probabilities(row,col);
                
                total_loss -= std::log(std::max(p,1e-7f));
                break;
            }
        }
    }

    
    // calculate error in output layer

    output.error = output.activation - targets;
    
    // dW = error * a_prev T
    // dB = error
    // error_prev = W T * error hadamard f'(z_prev)

    //accumulate output layer bias gradient across each row
    
    for(std::size_t col = 0; col < inputs.cols(); col++)
        for(std::size_t row = 0; row < output.size; row++)
        {
            const Scalar error = output.error(row,col);
            output.bias_gradient(row,0) += error;
        }
    
    //propagate backwards
    
    for(std::size_t layer_index = layers_ - 1;layer_index > 0;layer_index--)
    {
        //step backwards through previous layers
        
        const std::size_t previous_index = layer_index - 1;
        
        auto& current = layer_[layer_index];
        auto& previous = layer_[previous_index];
        
        current.weight_gradient = Matrix::multiply(current.error, previous.activation.transpose());
        
        //We need the weights coming from the input layer but we don't need errors or bias gradients of the input layer
        
        Matrix propagated_error = Matrix::multiply(current.weight.transpose(), current.error);
        
        if(previous_index == 0)
        {
            previous.error =std::move(propagated_error);
            
        }
        else
        {
            previous.error = Matrix::hadamard(propagated_error, previous.activation_function_.derivative(previous.activation));
            
            //accumulate output layer bias gradient across each row
            
            for(std::size_t row = 0; row < previous.size; row++)
            {
                Scalar sum = 0;
                
                for(std::size_t col = 0; col < inputs.cols(); col++)
                    sum += previous.error(row,col);
                
                previous.bias_gradient(row,0) += sum;
            }
        }
    }
        
    return total_loss / targets.cols();
}

void Neural::gradient_descent(std::size_t trainingSize, double learningRate)
{
    const Scalar scale = static_cast<Scalar>(learningRate) / static_cast<Scalar>(trainingSize);
    
    for (std::size_t j=1;j<layers_;j++) // loop through layers starting from second
    {
        layer_[j].bias -= layer_[j].bias_gradient * scale;
        layer_[j].weight -= layer_[j].weight_gradient * scale;
    }
}


void Neural::print_stats(std::ostream& stream)
    {
        std::size_t count;
        
        for(std::size_t layer_index = 1; layer_index < layers_;layer_index++)
        {
            auto& layer = layer_[layer_index];
            count = 0;
            double mean_weight_gradient = 0.0;
            double mean_activation = 0.0;
            
            for(std::size_t i = 0; i < layer.weight_gradient.size(); i++)
            {
                count++;
                mean_weight_gradient += std::abs(layer.weight_gradient.data()[i]);
            }
            mean_weight_gradient/=count;
            
            for(std::size_t i = 0; i < layer.activation.size(); i++)
                mean_activation += std::abs(layer.activation.data()[i]);
            
            mean_activation /= layer.activation.size();
            
            stream << "Layer " << layer_index << " mean weight gradient = " << mean_weight_gradient << " mean activation = " << mean_activation << std::endl;
        }
    }


Scalar Neural::cost_function(const std::vector<Scalar>& target)
{
    double cost=0;
    for(std::size_t i=0;i<target.size();i++)
        {
            cost+=((get_output(i)-target[i])*(get_output(i)-target[i]));
        }
    return cost;
}


void Neural::propagate()
{
    // Pre-activation z = W a_prev + b
    // Activation a = f(z)
    // input activations matrix a_prev: N x 1
    // weights matrix W: M x N
    // biases matrix b: M x 1
    
    for(std::size_t i=1;i<layers_;i++) // step through layers
    {
        auto &current = layer_[i];
        const auto& previous = layer_[i - 1];
        
        current.pre_activation = Matrix::multiply(current.weight, previous.activation) + current.bias;
        current.activation = current.activation_function_.activate(current.pre_activation);
    }
  
}

void Neural::propagateBatch(bool training)
{
    // Pre-activation z = W a_prev + b
    // Activation a = f(z)
    // input activations matrix a_prev: N x 1
    // weights matrix W: M x N
    // biases matrix b: M x 1
    
    const Scalar dropout = 0.3f;
    
    for(std::size_t i=1;i<layers_;i++) // step through layers
    {
        auto &current = layer_[i];
        const auto& previous = layer_[i - 1];
        
        current.pre_activation = Matrix::broadcastAdd(Matrix::multiply(current.weight, previous.activation), current.bias);
        if(i==1 && dropout > 0.0f)
        {
            current.activation = current.activation_function_.activate(current.pre_activation) *
            ((std::rand()/RAND_MAX) > dropout ?
            (1.0f/(1.0f-dropout)) :
            0.0f);
        }
        else
        {
            current.activation = current.activation_function_.activate(current.pre_activation);
        }
    }
  
}


Neural::Neural(std::vector<int> nodes_per_layer, const ActivationType hiddenActivationType, const ActivationType outputActivationType)
{
    configure(nodes_per_layer, hiddenActivationType, outputActivationType);
}

void Neural::configure(std::vector<int> nodes_per_layer, const ActivationType hiddenActivationType, const ActivationType outputActivationType)
{
    layers_ = nodes_per_layer.size();
    max_layers_ = 0;
    layer_.clear();
    layer_.reserve(layers_);
    
    std::cout << "Configuring neural with " << layers_ << " layers." << std::endl;
    
    for(std::size_t i=0;i<layers_;i++)
    {
        if(i < layers_ - 1)
        {
            layer_.emplace_back(activationFromType(hiddenActivationType));
        }
        else
        {
            layer_.emplace_back(activationFromType(outputActivationType));
        }
        
        if(nodes_per_layer[i]>max_layers_)max_layers_ = nodes_per_layer[i];
        layer_[i].size=nodes_per_layer[i];
        layer_[i].activation = Matrix(layer_[i].size,1);
        layer_[i].pre_activation = Matrix(layer_[i].size,1);
        layer_[i].error = Matrix(layer_[i].size,1);
        layer_[i].bias_gradient = Matrix(layer_[i].size,1);

        layer_[i].bias = Matrix(layer_[i].size,1);
        layer_[i].weight = Matrix(layer_[i].size,i>0 ? layer_[i-1].size : 1);
        layer_[i].weight_gradient = Matrix(layer_[i].size,i>0 ? layer_[i-1].size : 1);
        
        layer_[i].bias = layer_[i].bias.apply(
                                []
                                (Scalar)
                                {
                                    return((rand()%100)/100.0f+0.01f);
                                });

        layer_[i].weight = layer_[i].weight.apply(
                                []
                                (Scalar)
                                {
                                    return(0.1f - (Scalar)(rand()%100) / 500.0f);
                                });

        
        std::cout << "Layer: " << i << " Nodes: " << layer_[i].size << std::endl;
    }
    
    std::cout << "Largest Layer:" << max_layers_ << std::endl;
}

void Neural::zero_training_error()
{
    for(std::size_t i=1;i<layers_;i++)
    {
        layer_[i].zero_gradients(layer_[i-1].size);
    }
    
}

void Neural::set_input(data_set& data,std::size_t index)
{
    layer_[0].activation = Matrix(layer_[0].size, 1);
    for(std::size_t j=0;j<data.n_inputs();j++)
    {
        double input=data.get_data()[index].inputs[j];
        set_input(j,input);
        //std::cout << input << std::endl;
    }
}


void Neural::set_input(const std::vector<Scalar>& input_vector)
{
    layer_[0].activation = Matrix(layer_[0].size, 1);
    if(input_vector.size() != layer_[0].size)
    {
        std::cout << "Error: input vector size " << input_vector.size()
                  << " does not match net input layer size " << layer_[0].size
                  << std::endl;
        return;
    }

    for(std::size_t i = 0; i < input_vector.size(); i++)
    {
        set_input(i,input_vector[i]);
    }
}

void Neural::print_network(std::ostream& stream)
{
    print_dimensions(stream);
    print_weights(stream);
    print_biases(stream);
    print_values(stream);
}

void Neural::print_dimensions(std::ostream& stream)
{
    stream << "Layers: " << layers_ << std::endl;
    
    stream << "Nodes per layer: ";
    
    for(auto l: layer_)
        stream << l.size << " ";
    
    stream << std::endl;
    
}

void Neural::print_weights(std::ostream& stream)
{
    stream << "Weights:" << std::endl;
    
    for(std::size_t i=1;i<layers_;i++)
    {
        stream << "Layer " << i << std::endl;
        for(std::size_t j=0;j<layer_[i].size;j++)
        {
            stream << "Node " << j << std::endl;
            for(std::size_t k=0;k<layer_[i].weight.rows();k++)
            {
                stream << layer_[i].weight(j,k) << std::endl;
            }
                 
        }
            
    }
        
}

void Neural::print_biases(std::ostream& stream)
{
    stream << "Biases:" << std::endl;
    for(std::size_t j=0;j<max_layers_;j++)
    {
        for(std::size_t i=0;i<layers_;i++)
        {
            if(j<layer_[i].size)
            {
                stream << layer_[i].bias(j,0) << " ";
            }
            else
            {
                stream << "  ";
            }
           
        }
        stream << std::endl;
    }
}

void Neural::print_values(std::ostream& stream)
{
    stream << "Values:" << std::endl;
    for(std::size_t j=0;j<max_layers_;j++)
    {
        for(std::size_t i=0;i<layers_;i++)
        {
            if(j<layer_[i].size)
            {
                stream << layer_[i].activation(j,0) << " ";
            }
            else
            {
                stream << "  ";
            }
        }
        stream << std::endl;
    }
}

void Neural::print_errors(std::ostream& stream)
{
    stream << "Errors:" << std::endl;
    for(std::size_t j=0;j<max_layers_;j++)
    {
        for(std::size_t i=0;i<layers_;i++)
        {
            if(j<layer_[i].size)
            {
                stream << layer_[i].error(j,0) << " ";
            }
            else
            {
                stream << "  ";
            }
        }
        stream << std::endl;
    }
}

void Neural::print_training_errors(std::ostream& stream)
{
    stream << "Training Errors:" << std::endl;
    for(std::size_t j=0;j<max_layers_;j++)
    {
        for(std::size_t i=0;i<layers_;i++)
        {
            if(j<layer_[i].size)
            {
                stream << layer_[i].bias_gradient(j,0) << " ";
            }
            else
            {
                stream << "  ";
            }
        }
        stream << std::endl;
    }
}

void Neural::set_input(std::size_t node, Scalar value)
{
    layer_[0].activation(node,0) = value;
}

Scalar Neural::get_output(std::size_t node)
{
    return layer_[layers_-1].activation(node,0);
}


std::vector<Scalar> Neural::get_output()
{
    return std::vector<Scalar>(layer_[layers_-1].activation.data(),layer_[layers_-1].activation.data()+layer_[layers_-1].activation.size());
}


std::size_t Neural::find_highest_output(void)
{
    double max_value = 0;
    std::size_t max_node = 0;
    for(std::size_t i = 0; i < layer_[layers_-1].size; i++)
    {
            if(max_value<get_output(i))
            {
                max_value=get_output(i);
                max_node=i;
            }
    }
    return max_node;
}
