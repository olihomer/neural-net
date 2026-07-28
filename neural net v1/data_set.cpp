//
//  data_set.cpp
//  neural net v1
//
//  Created by Oliver Homer on 26/03/2024.
//

#include "data_set.hpp"
#include <fstream>
#include <iostream>


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
    
    myfile>>m_n_inputs;
    myfile>>m_n_outputs;
    myfile>>n_lines;
    
    std::cout << "Opened data file with " << m_n_inputs << " inputs and " << m_n_outputs << " outputs and " << n_lines << " lines." << std::endl;
    
        
    temp.inputs.resize(m_n_inputs);
    temp.outputs.resize(m_n_outputs);
    
    for(std::size_t j=0;j<n_lines;j++)
    {
            
        for(std::size_t i=0;i<m_n_inputs;i++)
        {
            myfile>>temp.inputs[i];
        }
        
       for(std::size_t o=0;o<m_n_outputs;o++)
        {
            myfile>>temp.outputs[o];
        }
 
        m_data.push_back(temp);
        
    }
    
    myfile.close();

     
    std::cout<<"Size of data vector: "<< m_data.size()<<std::endl;
    m_size = m_data.size();
}


void data_set::print_data(std::ostream& stream)
{
    stream << "Data:" << std::endl;
    
    for(auto d : m_data)
    {
        for(auto i:d.inputs)std::cout<<i<<" ";
        for(auto o:d.outputs)std::cout<<o<<" ";
        
        std::cout << std::endl;
    }
    
    std::cout << "Data end." << std::endl;
}
