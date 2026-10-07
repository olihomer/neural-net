//
//  EmnistData.hpp
//  neural net v1
//
//  Created by Oliver Homer on 06/10/2026.
//

#ifndef EmnistData_hpp
#define EmnistData_hpp

#include <cstddef>
#include "DataSet.hpp"
#include <string>
#include <fstream>
#include <sstream>

class EmnistData : public DataSet
{
public:
    EmnistData(const std::string& fileprefix, std::size_t size);
    static constexpr std::string_view labelSet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
    const int get_label (int index);

private:
    std::vector<int> label_;
};




#endif /* EmnistData_hpp */
