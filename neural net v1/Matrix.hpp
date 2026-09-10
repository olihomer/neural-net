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
#include <stdexcept>
#include <algorithm>


class Matrix
{
public:

    explicit Matrix(std::size_t rows, std::size_t cols, const std::vector<Scalar> data);
    explicit Matrix(std::size_t rows, std::size_t cols);

    
    std::size_t rows() const {return rows_;};
    std::size_t cols() const {return cols_;};
    std::size_t size() const {return rows_ * cols_;};
    std::vector<Scalar> getData() const {return data_;};
    
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
    
    template <typename Func>
    Matrix apply(Func func) const
    {
        Matrix result(rows_, cols_);
        
        for(std::size_t i=0; i<data_.size(); i++)
            result.data_[i]=func(data_[i]);
        
        return result;
    }
    
private:
    std::vector<Scalar> data_;
    std::size_t rows_;
    std::size_t cols_;
};


#endif /* Matrix_hpp */
