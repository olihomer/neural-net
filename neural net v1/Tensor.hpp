//
//  Tensor.hpp
//  neural net v1
//
//  Created by Oliver Homer on 13/09/2026.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include "NeuralTypes.hpp"
#include <vector>
#include <exception>

class Tensor{
public:
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape);
    
    [[nodiscard]] std::size_t rank() const{return shape_.size();};
    [[nodiscard]] std::size_t size() const{return data_.size();};
    [[nodiscard]] std::size_t dim(std::size_t axis) const{return shape_[axis];};
    
    [[nodiscard]] const std::vector<std::size_t>& shape() const{return shape_;};
    
    [[nodiscard]] Scalar* data(){return data_.data();};
    [[nodiscard]] const Scalar* data() const{return data_.data();};
    
    Scalar& operator()(std::size_t i);
    Scalar& operator()(std::size_t i, std::size_t j);
    Scalar& operator()(std::size_t i, std::size_t j, std::size_t k);
    Scalar& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l);

    void fill(Scalar value);
    void zero();
    void reshape(const std::vector<size_t>& shape);

private:
    std::vector<Scalar> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    
    void calculateStrides();
    
    [[nodiscard]]
    std::size_t offset(std::initializer_list<std::size_t> indices) const;
};


#endif /* Tensor_hpp */
