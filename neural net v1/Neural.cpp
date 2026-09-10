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
    set_input(input);
    propagate();
    const auto index = find_highest_output();
    return {index,get_output(index)};
}

void Neural::load(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    
    if(!file)
        throw std::runtime_error("Load file could not be opened");
    
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
    file.read(reinterpret_cast<char*>(&m_layers),sizeof(m_layers));
    
    m_layer.clear();

    //Activation functions, create layers
    for(std::size_t i = 0 ; i < m_layers; i++)
    {
        std::uint8_t activation;
        file.read(reinterpret_cast<char*>(&activation),sizeof(activation));
        m_layer.emplace_back(activationFromType(static_cast<ActivationType>(activation)));
        
        
    }
    
    m_max_layers = 0;
    
    //Load each layer
    for(auto &layer: m_layer)
    {
        layer.load(file);
        if(layer.size > m_max_layers)m_max_layers = layer.size;
    }
    
    if(!file)
        throw std::runtime_error("Failed while loading");
}

void Neural::save(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);
    
    if(!file)
        throw std::runtime_error("Save file could not be opened");
    
    //Magic number
    constexpr char MAGIC[] = {'N','N','E','T'};
    file.write(MAGIC, sizeof(MAGIC));
    
    //Version
    const uint8_t version = 3;
    file.write(reinterpret_cast<const char*>(&version),sizeof(version));
    
    //Number of layers
    file.write(reinterpret_cast<const char*>(&m_layers),sizeof(m_layers));
    
    //Activation functions
    for(std::size_t i = 0 ; i < m_layers; i++)
    {
        const std::uint8_t activation = (std::uint8_t)m_layer[i].activation_function_.type();
        file.write(reinterpret_cast<const char*>(&activation),sizeof(activation));
    }
    
    //Save each layer
    for(const auto &layer: m_layer)
    {
        layer.save(file);
    }
    
    if(!file)
        throw std::runtime_error("Failed while saving");
}



double Neural::trainBatch(const data_set &training_data, const std::span<const std::size_t> batch)
{
    //std::cout << "Training with " << training_data.size() << " data points." << std::endl;
    
    if(training_data.n_inputs() != m_layer[0].size || training_data.n_outputs() != m_layer[m_layers-1].size)
    {
        std::cout << "Error: training data size does not match net topology" << std::endl;
        std::cout << "Inputs: " << training_data.n_inputs() << " in data versus " << m_layer[0].size << " in net." << std::endl;
        std::cout << "Outputs: " << training_data.n_outputs() << " in data versus " << m_layer[m_layers-1].size << " in net." << std::endl;
        exit(1);
    }
    
    Scalar total_error = 0;
    
    zero_training_error();
    
    for(auto batch_index: batch)
    {
        const auto& d = training_data.get_data()[batch_index];
        auto& input = m_layer[0];
        auto& output = m_layer[m_layers-1];
        
        input.activation = Matrix(d.inputs.size(),1,d.inputs);
        
        propagate();
        
        // calculate error in output layer
        
        Matrix target_output(d.outputs.size(),1,d.outputs);
        output.error = output.activation - target_output;
    
        // dW = error * a_prev T
        // dB = error
        // error_prev = W T * error hadamard f'(z_prev)
        
        output.bias_gradient = output.bias_gradient + output.error;
        
        for(std::size_t layer_index = m_layers - 1;layer_index > 0;layer_index--)
        {
            //step backwards through previous layers
            
            const std::size_t previous_index = layer_index - 1;
            
            auto& current = m_layer[layer_index];
            auto& previous = m_layer[previous_index];
            
            current.weight_gradient = current.weight_gradient + Matrix::outer(current.error, previous.activation);
            
            //We need the weights coming from the input layer but we don't need errors or bias gradients of the input layer
            if (previous_index == 0)
            {
                    continue;
            }
            
            Matrix propagated_error = Matrix::multiply(current.weight.transpose(), current.error);
            previous.error = Matrix::hadamard(propagated_error, previous.activation_function_.derivative(previous.activation));
            previous.bias_gradient = previous.bias_gradient + previous.error;
        }
        
        
        total_error += cost_function(d.outputs);
        
    } //end of training loop;
    total_error /= (batch.size() * 2);
    
    return total_error;
}

void Neural::gradient_descent(std::size_t trainingSize, double learningRate)
{
    const Scalar scale = static_cast<Scalar>(learningRate) / static_cast<Scalar>(trainingSize);
    
    for (std::size_t j=1;j<m_layers;j++) // loop through layers starting from second
    {
        m_layer[j].bias = m_layer[j].bias - m_layer[j].bias_gradient * scale;
        m_layer[j].weight = m_layer[j].weight - m_layer[j].weight_gradient * scale;
    }
}


void Neural::print_stats(std::ostream& stream)
    {
        std::size_t count;
        
        for(std::size_t layer_index = 1; layer_index < m_layers;layer_index++)
        {
            auto& layer = m_layer[layer_index];
            count = 0;
            double mean_weight_gradient = 0.0;
            double mean_activation = 0.0;
            
                for(const auto& value: layer.weight_gradient.getData())
                {
                    count++;
                    mean_weight_gradient+=std::abs(value);
                }
            mean_weight_gradient/=count;
            
            for(const auto& A: layer.activation.getData()) mean_activation+=std::abs(A);
            
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
    
    for(std::size_t i=1;i<m_layers;i++) // step through layers
    {
        auto &current = m_layer[i];
        const auto& previous = m_layer[i - 1];
        
        current.pre_activation = Matrix::multiply(current.weight, previous.activation) + current.bias;
        current.activation = current.activation_function_.activate(current.pre_activation);
    }
  
}

Neural::Neural(std::vector<int> nodes_per_layer, const ActivationType hiddenActivationType, const ActivationType outputActivationType)
{
    configure(nodes_per_layer, hiddenActivationType, outputActivationType);
}

void Neural::configure(std::vector<int> nodes_per_layer, const ActivationType hiddenActivationType, const ActivationType outputActivationType)
{
    m_layers = nodes_per_layer.size();
    m_max_layers = 0;
    m_layer.clear();
    m_layer.reserve(m_layers);
    
    std::cout << "Configuring neural with " << m_layers << " layers." << std::endl;
    
    for(std::size_t i=0;i<m_layers;i++)
    {
        if(i < m_layers - 1)
        {
            m_layer.emplace_back(activationFromType(hiddenActivationType));
        }
        else
        {
            m_layer.emplace_back(activationFromType(outputActivationType));
        }
        
        if(nodes_per_layer[i]>m_max_layers)m_max_layers = nodes_per_layer[i];
        m_layer[i].size=nodes_per_layer[i];
        m_layer[i].activation = Matrix(m_layer[i].size,1);
        m_layer[i].pre_activation = Matrix(m_layer[i].size,1);
        m_layer[i].error = Matrix(m_layer[i].size,1);
        m_layer[i].bias_gradient = Matrix(m_layer[i].size,1);

        m_layer[i].bias = Matrix(m_layer[i].size,1);
        m_layer[i].weight = Matrix(m_layer[i].size,i>0 ? m_layer[i-1].size : 1);
        m_layer[i].weight_gradient = Matrix(m_layer[i].size,i>0 ? m_layer[i-1].size : 1);
        
        m_layer[i].bias = m_layer[i].bias.apply(
                                []
                                (Scalar)
                                {
                                    return((rand()%100)/100.0f+0.01f);
                                });

        m_layer[i].weight = m_layer[i].weight.apply(
                                []
                                (Scalar)
                                {
                                    return(0.1f - (Scalar)(rand()%100) / 500.0f);
                                });

        
        std::cout << "Layer: " << i << " Nodes: " << m_layer[i].size << std::endl;
    }
    
    std::cout << "Largest Layer:" << m_max_layers << std::endl;
}

void Neural::zero_training_error()
{
    for(std::size_t i=1;i<m_layers;i++)
    {
        m_layer[i].zero_gradients(m_layer[i-1].size);
    }
    
}

void Neural::set_input(data_set& data,std::size_t index)
{
    //std::cout << "Inputs set to: ";
    for(std::size_t j=0;j<data.n_inputs();j++)
    {
        double input=data.get_data()[index].inputs[j];
        set_input(j,input);
        //std::cout << input << std::endl;
    }
}


void Neural::set_input(const std::vector<Scalar>& input_vector)
{
    if(input_vector.size() != m_layer[0].size)
    {
        std::cout << "Error: input vector size " << input_vector.size()
                  << " does not match net input layer size " << m_layer[0].size
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
    stream << "Layers: " << m_layers << std::endl;
    
    stream << "Nodes per layer: ";
    
    for(auto l: m_layer)
        stream << l.size << " ";
    
    stream << std::endl;
    
}

void Neural::print_weights(std::ostream& stream)
{
    stream << "Weights:" << std::endl;
    
    for(std::size_t i=1;i<m_layers;i++)
    {
        stream << "Layer " << i << std::endl;
        for(std::size_t j=0;j<m_layer[i].size;j++)
        {
            stream << "Node " << j << std::endl;
            for(std::size_t k=0;k<m_layer[i].weight.rows();k++)
            {
                stream << m_layer[i].weight(j,k) << std::endl;
            }
                 
        }
            
    }
        
}

void Neural::print_biases(std::ostream& stream)
{
    stream << "Biases:" << std::endl;
    for(std::size_t j=0;j<m_max_layers;j++)
    {
        for(std::size_t i=0;i<m_layers;i++)
        {
            if(j<m_layer[i].size)
            {
                stream << m_layer[i].bias(j,0) << " ";
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
    for(std::size_t j=0;j<m_max_layers;j++)
    {
        for(std::size_t i=0;i<m_layers;i++)
        {
            if(j<m_layer[i].size)
            {
                stream << m_layer[i].activation(j,0) << " ";
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
    for(std::size_t j=0;j<m_max_layers;j++)
    {
        for(std::size_t i=0;i<m_layers;i++)
        {
            if(j<m_layer[i].size)
            {
                stream << m_layer[i].error(j,0) << " ";
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
    for(std::size_t j=0;j<m_max_layers;j++)
    {
        for(std::size_t i=0;i<m_layers;i++)
        {
            if(j<m_layer[i].size)
            {
                stream << m_layer[i].bias_gradient(j,0) << " ";
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
    m_layer[0].activation(node,0) = value;
}

Scalar Neural::get_output(std::size_t node)
{
    return m_layer[m_layers-1].activation(node,0);
}

std::size_t Neural::find_highest_output(void)
{
    double max_value = 0;
    std::size_t max_node = 0;
    for(std::size_t i = 0; i < m_layer[m_layers-1].size; i++)
    {
            if(max_value<get_output(i))
            {
                max_value=get_output(i);
                max_node=i;
            }
    }
    return max_node;
}
