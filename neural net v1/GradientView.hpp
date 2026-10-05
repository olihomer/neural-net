//
//  GradientView.hpp
//  neural net v1
//
//  Created by Oliver Homer on 05/10/2026.
//

#pragma once
#include <cstddef>
#include "NeuralTypes.hpp"

class GradientView
{
public:
    GradientView(Scalar* data, const std::size_t size, const std::size_t batchStride, const std::size_t dataStride)
    :data_(data),size_(size),batchStride_(batchStride),dataStride_(dataStride)
    {
        ;
    }
    
    Scalar operator()(std::size_t batchIndex, std::size_t dataIndex) const{return data_[batchIndex * batchStride_ + dataIndex * dataStride_];}
    
    std::size_t size() const{return size_;}
    
private:
    Scalar* data_;
    std::size_t size_;
    std::size_t batchStride_;
    std::size_t dataStride_;
};


