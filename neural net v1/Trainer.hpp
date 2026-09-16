//
//  Trainer.hpp
//  neural net v1
//
//  Created by Oliver Homer on 09/09/2026.
//

#ifndef Trainer_hpp
#define Trainer_hpp

#include <stdio.h>
#include "Neural.hpp"
#include "mnist_data.hpp"
#include "Trainable.hpp"

enum NetworkType
{
    MLP_NETWORK,
    CNN_CLASSIFIER_NETWORK
};

class Trainer
{
public:
    Trainer (Trainable &network, NetworkType networktype);
    
    void train(
               const data_set& data,
               std::size_t epochs,
               std::size_t batchSize,
               double learningRate,
               void(*progressCallback)(int32_t,double));
private:
    Trainable& network_;
    NetworkType networktype_;
    static std::random_device rd;
    static std::mt19937 rng;
    std::vector<std::size_t> order_;
};



#endif /* Trainer_hpp */
