//
//  AppEngine.hpp
//  neural net v1
//
//  Created by Oliver Homer on 31/08/2026.
//

#include "Neural.hpp"
#include "Trainable.hpp"
#include "CNN.hpp"
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

    ModelKind activeModel_ = ModelKind::CNN;
    
public:
    int runApp(void(*progress)(int32_t,double),
               int modelKind,
               int hiddenLayerSize,
               int epochs,
               int trainingExamples,
               int batchSize,
               double learningRate,
               int hiddenActivation,
               int outputActivation,
               int cnnConv1Channels,
               int cnnConv2Channels,
               int cnnClassifierHiddenLayerSize);
    void extracted(const std::vector<float> &vectorData);
    
    std::pair<int,float> sendRasterData(const float *data, std::size_t size);
    bool saveNetwork(const char *path);
    bool loadNetwork(const char *path);
    int activeModelKind() const;
    int cnnKernelOutputChannels(int layer) const;
    int cnnKernelInputChannels(int layer) const;
    int cnnKernelHeight(int layer) const;
    int cnnKernelWidth(int layer) const;
    float cnnKernelValue(int layer, int outputChannel, int inputChannel, int y, int x) const;
    int cnnInputChannels(int layer) const;
    int cnnInputHeight(int layer) const;
    int cnnInputWidth(int layer) const;
    float cnnInputValue(int layer, int channel, int y, int x) const;
    int cnnActivationChannels(int layer) const;
    int cnnActivationHeight(int layer) const;
    int cnnActivationWidth(int layer) const;
    float cnnActivationValue(int layer, int channel, int y, int x) const;
    
    void selectModel(ModelKind kind);
    Trainable& activeTrainable();
    
    AppEngine();
};
