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
        delete[] z0_;
        delete[] w_;
        delete[] d_;
        Common::ReleaseExpTable(expTable_);
    }
    void Init(uint32_t data_size, uint32_t weight_dim){
        weight_dim_ = weight_dim;
        data_size_ = data_size;
        z0_= new weight_t[data_size];
        w_= new weight_t[weight_dim];
        d_ = new weight_t[data_size];
        memset(w_, 0.0, sizeof(weight_t)*weight_dim);
        expTable_ = Common::InitExpTable();
        sigmoidTable_ = Common::InitSigmoidTable();
        //use norm distribution to initialize gradient
        /*std::default_random_engine generator(1);
        std::normal_distribution<weight_t > distribution(0,1.0);
        for (int i=0; i<weight_dim; ++i) {
            weight_t number = distribution(generator);
            w_[i] = number;
        }*/

    }

    weight_t fast_exp(weight_t t) const {
        assert(t <= 0);
        if(-t <= MIN_EXP){
            return expTable_[EXP_TABLE_SIZE-1];
        }
        weight_t exp_t = expTable_[(int)(t * EXP_TABLE_SIZE / MIN_EXP)];
        return  exp_t;

    }

    weight_t fast_sigmoid(weight_t t) const{
        if(t>=0.0){
          if(-t <= MIN_EXP){
              return sigmoidTable_[EXP_TABLE_SIZE-1];
          }
              return sigmoidTable_[(int)(-t * EXP_TABLE_SIZE / MIN_EXP)];
        }else{
          //printf("%d\n",t);
          return 1.0-fast_sigmoid(-t);
        }
    }

    weight_t sigmoid_func(weight_t t) const{
        if(t > 0.0){
            //return 1.0 / (1+std::exp(-t));
            weight_t s1 = 1.0/(1+fast_exp(-t));
            weight_t s2 = fast_sigmoid(t);
            printf("%20.20lf %20.20lf\n", s1, s2);
            return s1;
        }else{
            //weight_t exp_t = std::exp(t);
            weight_t exp_t = fast_exp(t);
            weight_t s1 = exp_t / (1+exp_t);
            weight_t s2 = fast_sigmoid(t);
            printf("%20.20lf %20.20lf\n", s1, s2);
            return s1;
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
    inline weight_t sparse_dot(const Dataset::FEATURE_NODE* a, const weight_t* b){
        weight_t c = 0.0;
        while (a->index != -1) {
            c += b[a->index] * a->value;
            a++;
        }
        return c;
    }
    weight_t CalcLoss(Dataset::FEATURE_NODE** X, weight_t* y, const weight_t* w, float alpha){
        weight_t f_ = 0.0;
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_NODE *x = X[data_index];
            z0_[data_index] = sparse_dot(x, w);
        }

        for (int data_index = 0; data_index < data_size_; ++data_index) {
            weight_t yz = fast_sigmoid(z0_[data_index] * y[data_index]);
            //D
            d_[data_index] = yz*(1-yz);
            f_ += -std::log(yz);
            z0_[data_index] = yz;
        }
        for (int dim = 0; dim < weight_dim_; ++dim) {
            f_ += alpha*w[dim]*w[dim];
        }
        return f_;
    }

    void CalcGradients(Dataset::FEATURE_NODE** X, weight_t* y, const weight_t* w, float alpha, weight_t* g){
        memset(g, 0, sizeof(weight_t)*weight_dim_);
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            weight_t yz = z0_[data_index];
            weight_t z_0 = (yz - 1.0) * y[data_index];
            z0_[data_index] = z_0;
        }

        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_NODE *x = X[data_index];
            weight_t z_0 = z0_[data_index];
            while (x->index != -1) {
                g[x->index] += x->value * z_0;
                x++;
            }
        }
        for (int dim = 0; dim < weight_dim_; ++dim) {
            g[dim] += alpha* w[dim];
        }
    }
    void CalcHv(Dataset::FEATURE_NODE** X, weight_t* s, weight_t* Hs){
        memset(Hs, 0, sizeof(weight_t)* weight_dim_);
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_NODE *x = X[data_index];
            weight_t xTs = d_[data_index]*sparse_dot(x, s);
            while (x->index != -1) {
                Hs[x->index] += x->value * xTs;
                x++;
            }
        }
        for (int i = 0; i < weight_dim_; ++i) {
            Hs[i] += s[i];
        }
    }

    void CalcDiagPreConditioner(Dataset::FEATURE_NODE** X, weight_t* M){
        memset(M, 1, sizeof(weight_t)*weight_dim_);
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            Dataset::FEATURE_NODE *x = X[data_index];
            while (x->index != -1) {
                M[x->index] += x->value * x->value * d_[data_index];
                x++;
            }
        }
    }

    void Prediction(Dataset::FEATURE_NODE** X, uint32_t num_data, std::vector<label_t>& out_y_pred){
        for (int i = 0; i < num_data; ++i) {
            Dataset::FEATURE_NODE* x = X[i];
            out_y_pred.push_back(predict(x));
        }

    }

    weight_t* weights(){
        return w_;
    }

private:
    label_t predict(Dataset::FEATURE_NODE*x){
        weight_t wx=0.0;
        while (x->index != -1) {
            wx += w_[x->index]*x->value;
            x++;
        }
        weight_t prob = fast_sigmoid(wx);
        return static_cast<label_t>(prob);
    }

private:
    uint32_t weight_dim_;
    uint32_t data_size_;
    weight_t* z0_;
    weight_t* w_;
    weight_t* d_;
    weight_t* expTable_;
    weight_t* sigmoidTable_;
};
#endif //FASTLINEAR_BINARY_OBJECTIVE_HPP
