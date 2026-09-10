//
//  test.cpp
//  neural net v1
//
//  Lightweight C++ tests for the learning project.
//

#include "ActivationFunction.hpp"
#include "Matrix.hpp"
#include "Neural.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr Scalar tolerance = 0.00001f;

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

void requireNear(Scalar actual, Scalar expected, const std::string& message)
{
    if(std::abs(actual - expected) > tolerance)
    {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

void requireMatrixValue(const Matrix& matrix, std::size_t row, std::size_t col, Scalar expected, const std::string& message)
{
    requireNear(matrix(row, col), expected, message);
}

void testMatrixConstruction()
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    require(matrix.rows() == 2, "matrix rows should match constructor");
    require(matrix.cols() == 3, "matrix cols should match constructor");
    require(matrix.size() == 6, "matrix size should be rows times cols");
    requireMatrixValue(matrix, 1, 2, 6.0f, "matrix should store values in row-major order");

    bool threw = false;
    try
    {
        Matrix invalid(2, 2, {1.0f, 2.0f, 3.0f});
    }
    catch(const std::runtime_error&)
    {
        threw = true;
    }

    require(threw, "matrix should reject data with the wrong element count");
}

void testMatrixArithmetic()
{
    Matrix lhs(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Matrix rhs(2, 2, {4.0f, 3.0f, 2.0f, 1.0f});

    Matrix sum = lhs + rhs;
    requireMatrixValue(sum, 0, 0, 5.0f, "matrix addition should add matching elements");
    requireMatrixValue(sum, 1, 1, 5.0f, "matrix addition should add matching elements");

    Matrix difference = lhs - rhs;
    requireMatrixValue(difference, 0, 0, -3.0f, "matrix subtraction should subtract matching elements");
    requireMatrixValue(difference, 1, 1, 3.0f, "matrix subtraction should subtract matching elements");

    Matrix scaled = lhs * 2.0f;
    requireMatrixValue(scaled, 0, 1, 4.0f, "matrix scalar multiplication should scale values");
    requireMatrixValue(scaled, 1, 0, 6.0f, "matrix scalar multiplication should scale values");

    lhs += rhs;
    requireMatrixValue(lhs, 0, 0, 5.0f, "matrix += should mutate matching elements");
    requireMatrixValue(lhs, 1, 1, 5.0f, "matrix += should mutate matching elements");

    lhs -= rhs;
    requireMatrixValue(lhs, 0, 0, 1.0f, "matrix -= should mutate matching elements");
    requireMatrixValue(lhs, 1, 1, 4.0f, "matrix -= should mutate matching elements");

    lhs *= 3.0f;
    requireMatrixValue(lhs, 0, 1, 6.0f, "matrix *= should scale in place");
    requireMatrixValue(lhs, 1, 0, 9.0f, "matrix *= should scale in place");
}

void testMatrixProducts()
{
    Matrix lhs(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Matrix rhs(3, 2, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    Matrix product = Matrix::multiply(lhs, rhs);
    require(product.rows() == 2 && product.cols() == 2, "matrix multiply should produce lhs rows by rhs cols");
    requireMatrixValue(product, 0, 0, 58.0f, "matrix multiply should calculate dot products");
    requireMatrixValue(product, 1, 1, 154.0f, "matrix multiply should calculate dot products");

    Matrix transposed = lhs.transpose();
    require(transposed.rows() == 3 && transposed.cols() == 2, "transpose should swap dimensions");
    requireMatrixValue(transposed, 2, 1, 6.0f, "transpose should move values to swapped coordinates");

    Matrix hadamard = Matrix::hadamard(
        Matrix(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}),
        Matrix(2, 2, {5.0f, 6.0f, 7.0f, 8.0f})
    );
    requireMatrixValue(hadamard, 0, 1, 12.0f, "hadamard should multiply matching elements");
    requireMatrixValue(hadamard, 1, 1, 32.0f, "hadamard should multiply matching elements");

    Matrix outer = Matrix::outer(
        Matrix(2, 1, {2.0f, 3.0f}),
        Matrix(3, 1, {4.0f, 5.0f, 6.0f})
    );
    require(outer.rows() == 2 && outer.cols() == 3, "outer product should produce lhs rows by rhs rows");
    requireMatrixValue(outer, 1, 2, 18.0f, "outer product should multiply each pair of vector values");
}

void testMatrixDimensionChecks()
{
    bool addThrew = false;
    try
    {
        Matrix lhs(2, 2);
        Matrix rhs(3, 2);
        lhs += rhs;
    }
    catch(const std::runtime_error&)
    {
        addThrew = true;
    }

    require(addThrew, "matrix += should reject mismatched dimensions");

    bool multiplyThrew = false;
    try
    {
        Matrix::multiply(Matrix(2, 2), Matrix(3, 2));
    }
    catch(const std::runtime_error&)
    {
        multiplyThrew = true;
    }

    require(multiplyThrew, "matrix multiply should reject incompatible dimensions");
}

void testNeuralSaveLoadRoundTrip()
{
    const std::string path = "/tmp/neural_net_v1_round_trip_test.nnet";
    const std::vector<Scalar> input = {0.25f, 0.75f};

    Neural original({2, 3, 2}, ActivationType::Sigmoid, ActivationType::Sigmoid);
    original.set_input(input);
    original.propagate();
    const Scalar originalOutput0 = original.get_output(0);
    const Scalar originalOutput1 = original.get_output(1);

    original.save(path);

    Neural loaded({1, 1}, ActivationType::Sigmoid, ActivationType::Sigmoid);
    loaded.load(path);
    loaded.set_input(input);
    loaded.propagate();

    requireNear(loaded.get_output(0), originalOutput0, "loaded network output 0 should match saved network");
    requireNear(loaded.get_output(1), originalOutput1, "loaded network output 1 should match saved network");

    std::remove(path.c_str());
}

void runTest(const std::string& name, void(*test)())
{
    test();
    std::cout << "[PASS] " << name << std::endl;
}

} // namespace

int main()
{
    try
    {
        runTest("Matrix construction", testMatrixConstruction);
        runTest("Matrix arithmetic", testMatrixArithmetic);
        runTest("Matrix products", testMatrixProducts);
        runTest("Matrix dimension checks", testMatrixDimensionChecks);
        runTest("Neural save/load round trip", testNeuralSaveLoadRoundTrip);
    }
    catch(const std::exception& error)
    {
        std::cerr << "[FAIL] " << error.what() << std::endl;
        return 1;
    }

    std::cout << "All tests passed." << std::endl;
    return 0;
}
