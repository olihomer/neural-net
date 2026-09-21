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

enum class ModelKind
{
    MLP,
    CNN
};

class AppEngine {
private:
    Neural mlp_;
    CNN cnn_;

    ModelKind activeModel_ = ModelKind::MLP;
    
public:
    int runApp(void(*progress)(int32_t,double), int hiddenLayerSize, int epochs, int trainingExamples, int batchSize, double learningRate, int hiddenActivation, int outputActivation);
    void extracted(const std::vector<float> &vectorData);
    
    std::pair<int,float> sendRasterData(const float *data, std::size_t size);
    bool saveNetwork(const char *path);
    bool loadNetwork(const char *path);
    
    void selectModel(ModelKind kind);
    Trainable& activeTrainable(){return (activeModel_ == ModelKind::CNN ? cnn_ : mlp_);};
    
    AppEngine();
    ~AppEngine(){std::cout << "Deconstructing Engine" << std::endl;};

};
