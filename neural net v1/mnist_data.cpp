//
//  mnist_data.cpp
//  neural net v1
//
//  Created by Oliver Homer on 09/04/2024.
//

#include "mnist_data.hpp"
#include <fstream>
#include <string>
#include <sstream>


mnist_data::mnist_data()
{
    data_set();
}


mnist_data::mnist_data(const std::string &filename, int size)
{
    data_set(); //call base ctor
    
    m_n_inputs = IN_DIM;;
    m_n_outputs = OUT_DIM;
    m_size = size;
    
    m_data.reserve(size);
    
    std::fstream myfile; // open data file
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
    
    int val;
    std::string line;
    
    
    std::getline(myfile,line); //discard header row

    
    for(std::size_t i=0;i<size;i++)
    {
        data temp_data;
        temp_data.inputs.reserve(m_n_inputs);
        temp_data.outputs.resize(m_n_outputs);
        
        std::getline(myfile,line); //read a single line
        std::stringstream ss(line);
        
        ss >> val; //get label
        m_label.push_back(val); //store label for convenience
        
        for(int j=0;j<10;j++) // set outputs based on label
        {
            temp_data.outputs[j]=0;
            if(val==j)temp_data.outputs[j]=1;
        }
        
        if(ss.peek() == ',') ss.ignore(); //skip leading comma
        
        while(ss >> val)
        {
            if(ss.peek() == ',') ss.ignore();
            temp_data.inputs.push_back(((double)val)/255); //store pixel data in inputs
        }
                
        m_data.push_back(temp_data);
        

    }
    
    for(std::size_t i=0;i<size;i++)
    {
        std::cout << m_label[i] << std::endl;
    }
    
    myfile.close();
}


void mnist_data::print_data(std::ostream& stream)
{
    stream << "Mnist Data. Size: " << m_size << std::endl;
    
  
    
    for(std::size_t i=0;i<m_size;i++)
    {
        stream << "Label: " << m_label[i] << std::endl;
        
        for(std::size_t j=0;j<OUT_DIM;j++)
        {
            stream << m_data[i].outputs[j];
        }
        
        
        for(std::size_t y=0;y<Y_DIM;y++)
        {
            for(std::size_t x=0;x<X_DIM;x++)
                if(m_data[i].inputs[y*Y_DIM+x]>PRINT_THRESHOLD)
                {
                    stream << "X ";
                }
                else
                {
                    stream << "  ";
                }
            stream << std::endl;
        }
        
        stream << std::endl;
    }
    
    stream << "Data end." << std::endl;
}

const int mnist_data::get_label(int index)
{
    return m_label[index];
}
