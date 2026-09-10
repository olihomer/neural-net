//
//  Matrix.cpp
//  neural net v1
//
//  Created by Oliver Homer on 10/09/2026.
//

#include "Matrix.hpp"
#include <iostream>

Matrix::Matrix(std::size_t rows, std::size_t cols, const std::vector<Scalar> data)
: rows_(rows), cols_(cols), data_(data)
{
    if(data.size() != rows * cols)throw std::runtime_error("Matrix data size mismatch");
}


Matrix::Matrix(std::size_t rows, std::size_t cols)
: rows_(rows), cols_(cols), data_(rows*cols)
{
    ;
}

Matrix Matrix::operator+(const Matrix& rhs) const
{
    if(cols_ != rhs.cols()  || rows_ != rhs.rows())throw std::runtime_error("Matrices cannot be added");

    Matrix m(rows_,cols_);
    for(std::size_t row = 0; row < m.rows(); row++)
        for(std::size_t col = 0; col < m.cols(); col++)
            m(row,col) = (*this)(row,col) + rhs(row,col);
    
    return m;
}

Matrix Matrix::operator-(const Matrix& rhs) const
{
    Matrix m(std::max(rows_,rhs.rows()),std::max(cols_,rhs.cols()));
    for(std::size_t row = 0; row < m.rows(); row++)
        for(std::size_t col = 0; col < m.cols(); col++)
            m(row,col) = (*this)(row,col) - rhs(row,col);
    
    return m;
}

Matrix Matrix::operator*(Scalar scalar) const
{
    Matrix m(rows_,cols_);
    for(std::size_t row = 0; row < m.rows(); row++)
        for(std::size_t col = 0; col < m.cols(); col++)
            m(row,col) = (*this)(row,col) * scalar;

    return m;
}

Matrix Matrix::multiply(const Matrix& lhs, const Matrix& rhs)
{
    if(lhs.cols() != rhs.rows())throw std::runtime_error("Matrices cannot be multiplied");
    
    Matrix m(lhs.rows(), rhs.cols());
    
    for(std::size_t row = 0; row < m.rows(); row++)
        for(std::size_t col = 0; col < m.cols(); col++)
            for(std::size_t i=0; i < lhs.cols(); i++)
                m(row,col) += lhs(row,i) * rhs (i,col);
    
    return m;
}

Matrix Matrix::hadamard(const Matrix& lhs, const Matrix& rhs)
{
    if(lhs.cols() != rhs.cols()  || lhs.rows() != rhs.rows())throw std::runtime_error("Matrices cannot be hadamarded");
    
    Matrix m(lhs.rows(), lhs.cols());
    
    for(std::size_t row = 0; row < m.rows(); row++)
        for(std::size_t col = 0; col < m.cols(); col++)
                m(row,col) = lhs(row,col) * rhs (row,col);
    
    return m;
}

Matrix Matrix::outer(const Matrix& lhs, const Matrix& rhs)
{
    if(lhs.cols()!=1 || rhs.rows()!=1)throw std::runtime_error("Outer product requires column vectors");
    Matrix m = Matrix::multiply(lhs,rhs.transpose());
    return m;
}


void Matrix::print() const
{
    for(std::size_t row = 0; row < rows_; row++)
    {
        for(std::size_t col = 0; col < cols_; col++)
        {
            std::cout << (*this)(row,col) << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::transpose() const
{
    Matrix m(cols_,rows_);
    
    for(std::size_t row = 0; row < rows_; row++)
        for(std::size_t col = 0; col < cols_; col++)
            m(col,row) = (*this)(row,col);
    
    return m;
}
