//
//  main.cpp
//  neural net v1
//
//  Created by Oliver Homer on 18/02/2024.
//

#include <iostream>
#include <fstream>
#include "neural.hpp"
#include "data_set.hpp"
#include "mnist_data.hpp"
#include <string>
#include <vector>



int LegacyApp(int argc, const char * argv[])
{
    neural net({784,10,10});
    
    
    
    //net.print_network(std::cout);
    
    mnist_data mnist_training_data("/Users/oliverhomer/Xcode/neural net v1/mnist_test.csv",100);
    mnist_training_data.print_data(std::cout);
    
    //training_data.print_data(std::cout);
    
    double total_error;
 
    for (int i=0;i<1000;i++)
    {
        total_error = net.train(mnist_training_data);
        net.gradient_descent();
        if(i % 20==0)
        {
            std::cout << i << " ";
            std::cout << "Total error: " << total_error << std::endl;
            net.print_stats(std::cout);
        }
    }
    
    
   for(int i=0;i<50;i++)
    {
        int guess = rand()%100;
        int guess_label = mnist_training_data.get_label(guess);
        std::cout << "Guess = " << guess_label;
        
        net.set_input(mnist_training_data, guess);
        net.propagate();
        
        std::cout << ". Net guessed " << net.find_highest_output() << " with value of " << net.get_output(net.find_highest_output()) << std::endl;
        if(net.find_highest_output()!=guess_label)std::cout<<"WRONG!"<<std::endl;
    }
    
    
    return 0;
}

