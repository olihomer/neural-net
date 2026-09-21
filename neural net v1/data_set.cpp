//
//  data_set.cpp
//  neural net v1
//
//  Created by Oliver Homer on 26/03/2024.
//

#include "data_set.hpp"
#include <fstream>
#include <iostream>



data_set::data_set(std::vector<Scalar> inputs, std::vector<Scalar> outputs) //constructor for singleton data item
{
    size_ = 1;
    data_.resize(size_);
    n_inputs_ = inputs.size();
    n_outputs_ = outputs.size();
    data_[0].inputs = inputs;
    data_[0].outputs = outputs;
}

data_set::data_set(const std::string& filename)
{
    std::fstream myfile;
    myfile.open(filename);
    if(myfile.is_open())
    {
        std::cout << "opened ok" << std::endl;
    }
    else
    {
        std::cout << "Error opening file";
        exit(1);
    }
    
    std::string s;
    data temp;
    int n_lines;
    
    myfile>>n_inputs_;
    myfile>>n_outputs_;
    myfile>>n_lines;
    
    std::cout << "Opened data file with " << n_inputs_ << " inputs and " << n_outputs_ << " outputs and " << n_lines << " lines." << std::endl;
    
        
    temp.inputs.resize(n_inputs_);
    temp.outputs.resize(n_outputs_);
    
    for(std::size_t j=0;j<n_lines;j++)
    {
            
        for(std::size_t i=0;i<n_inputs_;i++)
        {
            myfile>>temp.inputs[i];
        }
        
       for(std::size_t o=0;o<n_outputs_;o++)
        {
            myfile>>temp.outputs[o];
        }
 
        data_.push_back(temp);
        
    }
    
    myfile.close();

     
    std::cout<<"Size of data vector: "<< data_.size()<<std::endl;
    size_ = data_.size();
}


void data_set::print_data(std::ostream& stream)
{
    stream << "Data:" << std::endl;
    
    for(auto d : data_)
    {
        for(auto i:d.inputs)std::cout<<i<<" ";
        for(auto o:d.outputs)std::cout<<o<<" ";
        
        std::cout << std::endl;
    }
    
    std::cout << "Data end." << std::endl;
}
