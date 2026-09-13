//
//  Tensor.cpp
//  neural net v1
//
//  Created by Oliver Homer on 13/09/2026.
//

#include "Tensor.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>


Tensor::Tensor(const std::vector<size_t>& shape)
:shape_(shape), rank_(shape.size())
{
    strides_.resize(rank_);
    
    std::size_t data_size = 1;
    for(auto s: shape_)
        data_size *= s;
    
    data_.resize(data_size);
        
    calculateStrides();
}

void Tensor::calculateStrides()
{
    if(rank_ == 0)return;
    strides_[rank_-1] = 1;
   
    for(std::size_t i = rank_-1; i>0; i--)
        strides_[i-1] = strides_[i] * shape_[i];
}

void Tensor::print()
{
    std::cout << "Shape: ";
    for(auto s: shape_)
        std::cout << s << " ";
    std::cout << std::endl << "Strides: ";
    for(auto s: strides_)
        std::cout << s << " ";
    std::cout << std::endl << "Data size: " << size() << std::endl;
}

void Tensor::fill(Scalar value)
{
    std::fill(data_.begin(),data_.end(),value);
}

void Tensor::zero()
{
    fill(0.0f);
}

void Tensor::reshape(const std::vector<size_t>& shape)
{
    std::size_t data_size = 1;
    for(auto s: shape)
        data_size *= s;

    if(data_size!=size())throw std::runtime_error("Attempt to reshape tensor to different overall size");
    
    shape_ = shape;
    rank_ = shape.size();
    strides_.resize(rank_);
    calculateStrides();
}


std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const
{
    std::size_t offset = 0;
    if(indices.size()!=rank_)throw std::runtime_error("Offset call not compatible with tensor shape");
    for(std::size_t i=0; i<indices.size();i++)
        offset += strides_[i] * indices.begin()[i];
    return offset;
}


Scalar& Tensor::operator()(std::size_t i)
{
    return data_[offset({i})];
}

Scalar& Tensor::operator()(std::size_t i, std::size_t j)
{
    return data_[offset({i,j})];
}

Scalar& Tensor::operator()(std::size_t i, std::size_t j, std::size_t k)
{
    return data_[offset({i,j,k})];
}

Scalar& Tensor::operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l)
{
    return data_[offset({i,j,k,l})];
}
