//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_BINARY_OBJECTIVE_HPP
#define FASTLINEAR_BINARY_OBJECTIVE_HPP

#include "objective_function.h"
#include <cmath>
#include <random>
class BinaryObjective:public ObjectiveFunction{
public:
    BinaryObjective(){}
    ~BinaryObjective() {
        delete[] gradient_;
        delete[] z0_;
        delete[] w_;
    }
    void Init(uint32_t data_size, uint32_t weight_dim){
        weight_dim_ = weight_dim;
        data_size_ = data_size;
        gradient_ = new weight_t(weight_dim);
        z0_= new weight_t(data_size);
        w_= new weight_t(data_size);

        //use norm distribution to initialize gradient
        std::default_random_engine generator;
        std::normal_distribution<weight_t > distribution(0,1.0);
        for (int i=0; i<weight_dim; ++i) {
            weight_t number = distribution(generator);
            w_[i] = number;
        }


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
    yz = phi(y * z)
    z0 = (yz - 1) * y
    grad = X.T.dot(z0) + alpha * w

    z = X.dot(w)
    yz = phi(y * z)
    loss = log(1/yz)+alpha*w^2 = -log(yz)+alpha*w^2

    */
    void CalcGradients(Dataset::DATA_MAT& X, const Dataset::LABEL_VEC& y, const weight_t* w,  const float alpha){
        f_ = 0.0;
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_VALUE& k_v = X[data_index];
            double z = 0.0;
            for (int index = 0; index < k_v.size(); ++index) {
                //printf("%d_%d\n", data_index, k_v[index].first);
                assert(k_v[index].first>=0);
                z += w[k_v[index].first] * k_v[index].second;
            }
            weight_t yz = sigmoid_func(z*y[data_index]);
            f_+= -std::log(yz);
            weight_t z_0 = (yz-1.0)*y[data_index];
            z0_[data_index] = z_0;
        }
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_VALUE& k_v = X[data_index];
            for (int index = 0; index < k_v.size(); ++index) {
                gradient_[k_v[index].first] = k_v[index].second * z0_[data_index] + alpha* w[k_v[index].first];
            }
        }
        for (int i = 0; i < weight_dim_; ++i) {
            f_ += alpha*w[i]*w[i];
        }

    }

    weight_t* gradient(){
        return gradient_;
    }

    weight_t loss(){
        return f_;
    }

    weight_t* weights(){
        return w_;
    }

private:
    uint32_t weight_dim_;
    uint32_t data_size_;
    weight_t* gradient_;
    weight_t* z0_;
    weight_t f_;
    weight_t* w_;
};
#endif //FASTLINEAR_BINARY_OBJECTIVE_HPP
