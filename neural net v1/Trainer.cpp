//
//  Trainer.cpp
//  neural net v1
//
//  Created by Oliver Homer on 09/09/2026.
//

#include "Trainer.hpp"
#include <span>

std::random_device Trainer::rd;
std::mt19937 Trainer::rng(Trainer::rd());

Trainer::Trainer(Neural& network)
: network_(network)
{
}


void Trainer::train(mnist_data& data, std::size_t epochs, std::size_t batchSize, double learningRate, void(*progressCallback)(int32_t,double))
{
    const std::size_t requestedBatchSize = static_cast<std::size_t>(batchSize);
    const std::size_t trainingExampleCount = static_cast<std::size_t>(data.size());
    double total_error = 0;
 
    order_.resize(data.size());
    std::iota(order_.begin(),order_.end(),0);
    
    for(std::size_t i = 0; i < epochs; i++)
    {
        double epoch_error = 0;
        std::size_t batches = 0;
        
        std::shuffle(order_.begin(),order_.end(), rng);

        for(std::size_t training_index = 0; training_index < trainingExampleCount; training_index += requestedBatchSize)
        {
            const std::size_t currentBatchSize = std::min(requestedBatchSize, trainingExampleCount - training_index);
            
            std::span<std::size_t> batch(order_.data() + training_index,currentBatchSize);
            epoch_error += network_.trainBatch(data, batch);
            network_.gradient_descent(currentBatchSize, learningRate);
            batches++;
        }

        total_error = epoch_error / double(batches);

        if((i+1) % 50 == 0 || i == epochs - 1)
        {
            std::cout << i << " ";
            std::cout << "Total error: " << total_error << std::endl;
            network_.print_stats(std::cout);
            progressCallback(int(i),total_error);
        }
        
    }
   
}
