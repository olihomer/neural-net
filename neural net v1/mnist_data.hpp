//
//  mnist_data.hpp
//  neural net v1
//
//  Created by Oliver Homer on 09/04/2024.
//

#ifndef mnist_data_hpp
#define mnist_data_hpp

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "data_set.hpp"


class mnist_data : public data_set
{
public:
    mnist_data();
    mnist_data(const std::string &filename, int size);
    void print_data(std::ostream& stream);
    const int get_label (int index);
    
    static const int X_DIM = 28;
    static const int Y_DIM = 28;
    static const int IN_DIM = X_DIM * Y_DIM;
    static const int OUT_DIM = 10;
    static constexpr float PRINT_THRESHOLD = 50.0f/255.0f;

private:
    std::vector<int> m_label;
};






#endif /* mnist_data_hpp */
