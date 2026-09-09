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
#include <numeric>
#include <random>
#include <algorithm>

std::random_device mnist_data::rd;
std::mt19937 mnist_data::rng(mnist_data::rd());

mnist_data::mnist_data()
{
    data_set();
}

mnist_data::mnist_data(const std::string &filename, int size)
{
    data_set(); //call base ctor
    
    m_order.resize(size);
    std::iota(m_order.begin(),m_order.end(),0);
    
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
    
    myfile.close();
}


void mnist_data::shuffle()
{
    std::shuffle(m_order.begin(),m_order.end(), rng);
}

const std::size_t mnist_data::get_order(const std::size_t index) const
{
    return m_order[index];
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


std::vector<float> mnist_data::preProcess(std::vector<float> rasterInput)
{
    std::vector<float> output;
    output.resize(28 * 28);
    
    const std::size_t sizeofside = static_cast<std::size_t>(std::sqrt(rasterInput.size()));
    if(rasterInput.empty() || sizeofside * sizeofside != rasterInput.size())
    {
        std::cout << "Error: input raster size " << rasterInput.size() << " is not square" << std::endl;
        return output;
    }
    
    std::size_t minX = sizeofside - 1;
    std::size_t maxX = 0;
    std::size_t minY = sizeofside - 1;
    std::size_t maxY = 0;
    
    const float threshold = 5.0f/255.0f;
    
    //establish bounding box and centre of mass
    
    float weighted_x = 0;
    float weighted_y = 0;
    float total_weight = 0;
    
    for(std::size_t y = 0; y < sizeofside; y++)
        for(std::size_t x = 0; x < sizeofside; x++)
        {
            float pixel = rasterInput[y*sizeofside+x];
            weighted_x += x * pixel;
            weighted_y += y * pixel;
            total_weight += pixel;
            
            if(pixel > threshold)
            {
                minX = x < minX ? x : minX;
                minY = y < minY ? y : minY;
                maxX = x > maxX ? x : maxX;
                maxY = y > maxY ? y : maxY;
            }
        }
    
    std::cout << "X range: " << minX << " to " << maxX << std::endl;
    std::cout << "Y range: " << minY << " to " << maxY << std::endl;
    
    //Blank drawing, return blank
    if (minX > maxX || minY > maxY) return output;
    
    //establish dimensions, fixed step to maintain aspect and centre
    
    const float width = float(maxX -  minX + 1);
    const float height = float(maxY - minY + 1);
    
    const float step = std::max(width, height) / 20.0f;
    
    const float centreX = (float(minX) + float(maxX)) / 2.0f;
    const float centreY = (float(minY) + float(maxY)) / 2.0f;
    
    const float weighted_centreX = weighted_x / total_weight;
    const float weighted_centreY = weighted_y / total_weight;
    
    std::cout << "Centre: " << centreX << "," << centreY << " Weighted: " << weighted_centreX << "," << weighted_centreY << std::endl;
 
    //bilinear interpolation
    
    for(std::size_t yDest = 0; yDest < 28; yDest++)
        for(std::size_t xDest = 0; xDest < 28; xDest++)
        {
            const float xSrc = weighted_centreX + (float(xDest) - 13.5f) * step;
            int x1 = std::floor(xSrc);
            int x2 = std::ceil(xSrc);
            float dx = xSrc - x1;
            
            const float ySrc = weighted_centreY + (float(yDest) - 13.5f) * step;
            int y1 = std::floor(ySrc);
            int y2 = std::ceil(ySrc);
            float dy = ySrc - y1;
            
            //leave black if virtual square extends beyond original
            if(xSrc < 0.0f || xSrc > float(sizeofside - 1) || ySrc < 0.0f || ySrc > float(sizeofside - 1)) continue;
            
            output[yDest * 28 + xDest] =    (1-dx)*(1-dy)*rasterInput[y1*sizeofside+x1] +
                                            dx*(1-dy)*rasterInput[y1*sizeofside+x2] +
                                            (1-dx)*dy*rasterInput[y2*sizeofside+x1] +
                                            dx*dy*rasterInput[y2*sizeofside+x2];
        }
    
    return output;
    
}
