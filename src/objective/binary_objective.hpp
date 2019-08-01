//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_BINARY_OBJECTIVE_HPP
#define FASTLINEAR_BINARY_OBJECTIVE_HPP

#include "objective_function.h"
#include <cmath>
class BinaryObjective:public ObjectiveFunction{
public:
    BinaryObjective(){}
    ~BinaryObjective() {
        delete[] weights_;

    }
    void Init(uint32_t data_size, uint32_t weight_dim){
        weight_dim_ = weight_dim;
        data_size_ = data_size;
        weights_ = new weight_t(weight_dim);
        z0_= new weight_t(data_size);
    }

    weight_t sigmoid_func(weight_t t) const{
        if(t > 0.0){
            return 1.0 / (1+std::exp(-t));
        }else{
            weight_t exp_t = std::exp(t);
            return exp_t / (1+exp_t);
        }
    }
    /*
    z = X.dot(w)
    z = phi(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
    */
    void GetGradients(Dataset::DATA_MAT& X, const Dataset::LABEL_VEC& y, float alpha, weight_t** w) const{
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_VALUE& k_v = X[data_index];
            double z = 0.0;
            for (int index = 0; index < k_v.size(); ++index) {
                z += w[k_v[index].first] * k_v[index].second;
            }
            weight_t z_0 = (sigmoid_func(z*y[data_index])-1.0)*y[data_index];
            z0_[data_index] = z_0;
        }
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_VALUE& k_v = X[data_index];
            for (int index = 0; index < k_v.size(); ++index) {
                weights_[k_v[index].first] = k_v[index].second * z0_[data_index] + alpha* weights_[k_v[index].first];
            }
        }
        *w = weights_;

    }
private:
    uint32_t weight_dim_;
    uint32_t data_size_;
    weight_t* weights_;
    weight_t* z0_;
};
#endif //FASTLINEAR_BINARY_OBJECTIVE_HPP
