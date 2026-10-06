//
//  Trainable.hpp
//  neural net v1
//
//  Created by Oliver Homer on 16/09/2026.
//

#ifndef Trainable_hpp
#define Trainable_hpp

#include <stdio.h>
#include <span>
#include <iostream>
#include "DataSet.hpp"
#include <cstddef>

class Trainable
{
public:
    virtual ~Trainable() = default;
    virtual void gradient_descent(std::size_t trainingSize, double learningRate) = 0;
    virtual double trainBatch(const DataSet& training_data, const std::span<const std::size_t> batch) = 0;
    virtual void print_stats(std::ostream& ostream) = 0;
    virtual void resetBeta() = 0;
};


#endif /* Trainable_hpp */
