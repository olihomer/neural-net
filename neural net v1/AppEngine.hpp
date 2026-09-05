//
//  AppEngine.hpp
//  neural net v1
//
//  Created by Oliver Homer on 31/08/2026.
//

#include "Neural.hpp"
#include <functional>

class AppEngine {
    Neural net;
public:
    int runApp(void(*progress)(int32_t,double));
    void sendRasterData(const float *data, std::size_t size);
    AppEngine();
    ~AppEngine();

};
