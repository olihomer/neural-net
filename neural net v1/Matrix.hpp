//
//  Matrix.hpp
//  neural net v1
//
//  Created by Oliver Homer on 10/09/2026.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include "NeuralTypes.hpp"
#include <vector>


class Matrix
{
public:

    explicit Matrix(const std::size_t rows, const std::size_t cols, const std::vector<Scalar> data);
    
    std::size_t rows() const {return rows_;};
    std::size_t cols() const {return cols_;};
    std::size_t size() const {return rows_ * cols_;};
    
    Scalar& operator()(std::size_t row, std::size_t col){return data_[row*cols_+col];};
    const Scalar& operator()(std::size_t row, std::size_t col) const {return data_[row*cols_+col];};

    Matrix operator+(const Matrix&) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator*(Scalar scalar) const;
    
    static Matrix multiply(const Matrix&, const Matrix&);
    static Matrix hadamard(const Matrix&, const Matrix&);
    static Matrix outer(const Matrix&, const Matrix&);

    Matrix transpose() const;
    
    void print() const;
    
private:
    std::vector<Scalar> data_;
    const std::size_t rows_;
    const std::size_t cols_;
};


#endif /* Matrix_hpp */
