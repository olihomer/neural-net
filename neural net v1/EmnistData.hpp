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

private:
    
};




#endif /* EmnistData_hpp */
