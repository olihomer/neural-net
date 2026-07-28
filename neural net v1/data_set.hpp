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

struct data
{
    std::vector<double> inputs;
    std::vector<double> outputs;
};

class data_set
{
public:
    //Constructors
    data_set(){};
    data_set(const std::string& filename);
    //Getter
    const std::vector<data>& get_data() const {return m_data;};
    const std::size_t n_inputs() const {return m_n_inputs;};
    const std::size_t n_outputs() const {return m_n_outputs;};
    //Debug
    void print_data(std::ostream& stream);
    
protected:
    std::vector<data> m_data;
    std::size_t m_n_inputs;
    std::size_t m_n_outputs;
    std::size_t m_size;
};




#endif /* data_set_hpp */
