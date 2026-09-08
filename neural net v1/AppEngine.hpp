//
//  AppEngine.hpp
//  neural net v1
//
//  Created by Oliver Homer on 31/08/2026.
//

#include "Neural.hpp"
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

class AppEngine {
private:
    Neural net;

public:
    int runApp(void(*progress)(int32_t,double), int hiddenLayerSize, int epochs, int trainingExamples, double learningRate, int hiddenActivation, int outputActivation);
    std::pair<int,float> sendRasterData(const float *data, std::size_t size);
    bool saveNetwork(const char *path);
    bool loadNetwork(const char *path);
    std::vector<float> preProcess(std::vector<float>);
    AppEngine();
    ~AppEngine();

};
