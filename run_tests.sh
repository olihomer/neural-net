#!/bin/sh

set -eu

clang++ -std=gnu++20 \
    -I"neural net v1" \
    "neural net v1/test.cpp" \
    "neural net v1/Matrix.cpp" \
    "neural net v1/Neural.cpp" \
    "neural net v1/NeuronLayer.cpp" \
    "neural net v1/ActivationFunction.cpp" \
    "neural net v1/data_set.cpp" \
    -o /tmp/neural_net_v1_tests

/tmp/neural_net_v1_tests
