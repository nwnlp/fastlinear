//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_BINARY_OBJECTIVE_HPP
#define FASTLINEAR_BINARY_OBJECTIVE_HPP

#include "objective_function.h"
#include <cmath>
#include <random>
#include <assert.h>
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
        gradient_ = new weight_t[weight_dim];
        z0_= new weight_t[data_size];
        w_= new weight_t[weight_dim];
        memset(w_, 0.1, sizeof(weight_t)*weight_dim);
        //use norm distribution to initialize gradient
        /*std::default_random_engine generator(1);
        std::normal_distribution<weight_t > distribution(0,1.0);
        for (int i=0; i<weight_dim; ++i) {
            weight_t number = distribution(generator);
            w_[i] = number;
        }*/


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
    void CalcGradients(const Dataset::DATA_MAT& X, const Dataset::LABEL_VEC& y, const weight_t* w,  const float alpha){
        clock_t start = clock();
        f_ = 0.0;
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            //Dataset::FEATURE_VALUE& k_v = X[data_index];
            weight_t z = 0.0;
            for (int index = 0; index < X[data_index].size(); ++index) {
                //printf("%d_%d\n", data_index, k_v[index].first);
                //assert(X[data_index][index].first>=0);
                z += w[X[data_index][index].first] * X[data_index][index].second;
            }
            weight_t yz = sigmoid_func(z*y[data_index]);
            f_+= -std::log(yz);
            weight_t z_0 = (yz-1.0)*y[data_index];
            z0_[data_index] = z_0;
        }

        for (int dim = 0; dim < weight_dim_; ++dim) {
            gradient_[dim]=0.0;
        }
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            //Dataset::FEATURE_VALUE& k_v = X[data_index];
            for (int index = 0; index < X[data_index].size(); ++index) {
                //Log::Info("%d_%d\n", data_index, X[data_index][index].first);
                gradient_[X[data_index][index].first] += X[data_index][index].second * z0_[data_index];
            }
        }
        for (int dim = 0; dim < weight_dim_; ++dim) {
            gradient_[dim] += alpha* w[dim];
        }

        for (int i = 0; i < weight_dim_; ++i) {
            f_ += alpha*w[i]*w[i];
        }
        clock_t end = clock();
        printf("grad time=%f\n",(float)(end-start)*1000/CLOCKS_PER_SEC);
    }



    void Prediction(const Dataset::DATA_MAT& X, std::vector<label_t>& out_y_pred){

        for (int i = 0; i < X.size(); ++i) {
            out_y_pred.push_back(predict(X[i]));
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
    label_t predict(const Dataset::FEATURE_VALUE& feature_values){
        weight_t wx=0.0;
        for (int i = 0; i < feature_values.size(); ++i) {
            wx += w_[feature_values[i].first]*feature_values[i].second;
        }
        weight_t prob = sigmoid_func(wx);
        return static_cast<label_t>(prob);
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
