//
//  neural.cpp
//  neural net v1
//
//  Created by Oliver Homer on 27/02/2024.
//

#include "Neural.hpp"
#include <iostream>
#include <cmath>


double Neural::train(const data_set &training_data)
{
    //std::cout << "Training with " << training_data.size() << " data points." << std::endl;
    
    if(training_data.n_inputs() != m_layer[0].size || training_data.n_outputs() != m_layer[m_layers-1].size)
    {
        std::cout << "Error: training data size does not match net topology" << std::endl;
        std::cout << "Inputs: " << training_data.n_inputs() << " in data versus " << m_layer[0].size << " in net." << std::endl;
        std::cout << "Outputs: " << training_data.n_outputs() << " in data versus " << m_layer[m_layers-1].size << " in net." << std::endl;
        exit(1);
    }
    
    
    double total_error = 0;
    
    zero_training_error();
    
    for(const auto& d : training_data.get_data()) //step through training set
    {
        
        for(std::size_t input=0;input<training_data.n_inputs();input++)
        {
            set_input(input, d.inputs[input]);
            //std::cout << d.inputs[input];
        }
        //std::cout << std::endl;
        propagate();
        
        // calculate error in output layer
        
        for(std::size_t i=0;i<m_layer[m_layers-1].size;i++)
        {
            //m_layer[m_layers-1].error[i] = (get_output(i)-d.o[i]) * sigmoid_prime(m_layer[m_layers-1].weighted_input[i]);  QUADRATIC
            m_layer[m_layers-1].error[i] = (get_output(i)-d.outputs[i]); //cross-entropy loss with sigmoid simplification
            m_layer[m_layers-1].bias_gradient[i] += m_layer[m_layers-1].error[i];
        }
        
        for(std::size_t layer_index = m_layers - 1;layer_index > 0;layer_index--)
        {
            //step backwards through previous layers
            
            const std::size_t previous_index = layer_index - 1;
            
            auto& current = m_layer[layer_index];
            auto& previous = m_layer[previous_index];
            
            //Accumulate gradients for weights entering this layer. Process a complete weight row before moving on
            for(std::size_t k = 0; k < current.size; k++)
            {
                auto &gradients = current.weight_gradient[k];
                const double error = current.error[k];
                
                for(std::size_t i = 0; i < previous.size; i++)
                {
                    gradients[i] += error * previous.activation[i];
                }
            }
                
            //We need the weights coming from the input layer but we don't need errors or bias gradients of the input layer
            if (previous_index == 0)
            {
                    continue;
            }
            
            //Calculate the previous layer's errors and accumulate its bias gradients
            for(std::size_t i = 0; i < previous.size; i++)
            {
            
                double propagated_error = 0.0;
                
                for (std::size_t k=0; k < current.size; k++) //step through forward connected nodes
                {
                    propagated_error += current.error[k] * current.weight[k][i];
                }
 
                
                previous.error[i] = propagated_error * previous.activation_function_.derivative(m_layer[previous_index].pre_activation[i]);
                
                previous.bias_gradient[i] += previous.error[i];
              
            }
        }
        
        total_error += cost_function(d.outputs);
        
        //std::cout << "Out:" << get_output(0) << std::endl;
        //std::cout << "Delta: " << get_output(0)-d.o << std::endl;
        //std::cout << "Training error: " << m_layer[m_layers-1].training_error[0] << std::endl;
        
    } //end of training loop;
    total_error /= (training_data.get_data().size()*2);
    
    return total_error;
    
}

void Neural::gradient_descent()
{
    for (std::size_t j=1;j<m_layers;j++) // loop through layers starting from second
    {
        for(std::size_t i=0;i<m_layer[j].size;i++) // loop through nodes
        {
            m_layer[j].bias[i] -= m_layer[j].bias_gradient[i] * learning_rate;
            for(std::size_t k=0;k<m_layer[j-1].size;k++) // loop through connected nodes
            {
                m_layer[j].weight[i][k]-=m_layer[j].weight_gradient[i][k] * learning_rate;

            }
        }
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
            
            for(const auto& vector_of_doubles: layer.weight_gradient)
                for(auto value: vector_of_doubles)
                {
                    count++;
                    mean_weight_gradient+=std::abs(value);
                }
            mean_weight_gradient/=count;
            
            for(double A: layer.activation) mean_activation+=std::abs(A);
            
            mean_activation /= layer.activation.size();
                
            
            
            stream << "Layer " << layer_index << " mean weight gradient = " << mean_weight_gradient << " mean activation = " << mean_activation << std::endl;
        }
    }


double Neural::cost_function(const std::vector<double>& target)
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
    
    double z;
    
    for(std::size_t i=1;i<m_layers;i++) // step through layers
    {
        for(std::size_t j=0;j<m_layer[i].size;j++)  // step through nodes
        {
            z=0;
            for(std::size_t k=0;k<m_layer[i].weight[j].size();k++)    // step through connections
            {
                z+=m_layer[i-1].activation[k]*m_layer[i].weight[j][k];
            }
            z+=m_layer[i].bias[j];
            m_layer[i].pre_activation[j]=z;
            m_layer[i].activation[j]=m_layer[i].activation_function_.activate(z);
        }
    }
}

void Neural::propogate()
{
    propagate();
}


Neural::Neural(std::vector<int> nodes_per_layer)
{
    m_layers=(int)nodes_per_layer.size();
    
    std::cout << "Constructing neural with " << m_layers << " layers." << std::endl;
 
    m_layer.resize(m_layers);
    
    for(std::size_t i=0;i<m_layers;i++)
    {
        if(nodes_per_layer[i]>m_max_layers)m_max_layers = nodes_per_layer[i];
        m_layer[i].size=nodes_per_layer[i];
        m_layer[i].activation.resize(m_layer[i].size);
        m_layer[i].pre_activation.resize(m_layer[i].size);
        m_layer[i].error.resize(m_layer[i].size);
        m_layer[i].bias_gradient.resize(m_layer[i].size);

        m_layer[i].bias.resize(m_layer[i].size);
        m_layer[i].weight.resize(m_layer[i].size);
        m_layer[i].weight_gradient.resize(m_layer[i].size);
        
        for(auto &b:m_layer[i].bias)
        {
            b=(double)(rand()%100)/100+0.01;
        }
        
        if(i>0)
        {
            m_layer[i].resize_for_previous(m_layer[i-1].size);
            for(std::size_t j=0;j<m_layer[i].size;j++)
            {
                for(auto &w:m_layer[i].weight[j])
                {
                    w = 0.1 - (double)(rand()%100) / 500;
                }
            }
        }
        
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


void Neural::set_input(std::vector<float> input_vector)
{
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
            for(std::size_t k=0;k<m_layer[i].weight[j].size();k++)
            {
                stream << m_layer[i].weight[j][k] << std::endl;
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
                stream << m_layer[i].bias[j] << " ";
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
                stream << m_layer[i].activation[j] << " ";
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
                stream << m_layer[i].error[j] << " ";
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
                stream << m_layer[i].bias_gradient[j] << " ";
            }
            else
            {
                stream << "  ";
            }
        }
        stream << std::endl;
    }
}

void Neural::set_input(std::size_t node, double value)
{
    m_layer[0].activation[node] = value;
}

double Neural::get_output(std::size_t node)
{
    return m_layer[m_layers-1].activation[node];
}

void Neural::set_bias(std::size_t layer, std::vector<double> bias)
{
    if(bias.size() == m_layer[layer].bias.size())
    {
        m_layer[layer].bias = bias;
    }
    else
    {
        std::cout << "Error: bias data wrong size" << std::endl;
    }
}

void Neural::set_weight(std::size_t layer, std::size_t node, std::vector<double> weight)
{
    if(weight.size() == m_layer[layer].weight[node].size())
    {
        m_layer[layer].weight[node]=weight;
    }
    else
    {
        std::cout << "Error: weight data wrong size" << std::endl;
    }
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

