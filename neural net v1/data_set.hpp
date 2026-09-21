//
//  data_set.hpp
//  neural net v1
//
//  Created by Oliver Homer on 26/03/2024.
//

#ifndef data_set_hpp
#define data_set_hpp

#include <stdio.h>
#include <vector>
#include "NeuralTypes.hpp"
#include <string>
#include <iosfwd>

struct data
{
    std::vector<Scalar> inputs;
    std::vector<Scalar> outputs;
};

class data_set
{
public:
    //Constructors
    data_set(){};
    data_set(std::vector<Scalar> inputs, std::vector<Scalar> outputs);
    data_set(const std::string& filename);
    //Getter
    const std::vector<data>& get_data() const {return data_;};
    const std::size_t n_inputs() const {return n_inputs_;};
    const std::size_t n_outputs() const {return n_outputs_;};
    const std::size_t size() const {return size_;};
    //Debug
    void print_data(std::ostream& stream);
    
protected:
    std::vector<data> data_;
    std::size_t n_inputs_;
    std::size_t n_outputs_;
    std::size_t size_;
};




#endif /* data_set_hpp */
