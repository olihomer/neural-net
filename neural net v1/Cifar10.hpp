//
//  Cifar10.hpp
//  neural net v1
//
//  Created by Oliver Homer on 07/10/2026.
//

#ifndef Cifar10_hpp
#define Cifar10_hpp

#include <cstddef>
#include "DataSet.hpp"
#include <string>
#include <fstream>
#include <sstream>

class Cifar10 : public DataSet
{
public:
    Cifar10(const std::string& filename, std::size_t size);
    static constexpr std::array<std::string_view,10> labelSet = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
    const int get_label (int index);

private:
    std::vector<int> label_;
};


#endif /* Cifar10_hpp */
