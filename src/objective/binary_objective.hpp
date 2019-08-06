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
        Common::ReleaseExpTable(expTable_);
    }
    void Init(uint32_t data_size, uint32_t weight_dim){
        weight_dim_ = weight_dim;
        data_size_ = data_size;
        gradient_ = new weight_t[weight_dim];
        z0_= new weight_t[data_size];
        w_= new weight_t[weight_dim];
        memset(w_, 0.1, sizeof(weight_t)*weight_dim);
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
    void CalcGradients(Dataset::FEATURE_NODE** X, weight_t* y, const weight_t* w, float alpha){
        clock_t start = clock();
        weight_t max_z = -10000;
        f_ = 0.0;
        OMP_INIT_EX();
        #pragma omp parallel for schedule(static)
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            const Dataset::FEATURE_NODE* x = X[data_index];
            weight_t z = 0.0;
            //x.dot(w)
            while (x->index != -1) {
                z += w[x->index] * x->value;
                x++;
            }
            max_z = std::max(std::fabs(z), max_z);
            //yz = phi(y * z)
            weight_t yz = fast_sigmoid(z*y[data_index]);
            OMP_LOOP_EX_BEGIN();
            f_+= -std::log(yz);
            OMP_LOOP_EX_END();
            //f_+= -logTable_[(int)(yz * LOG_TABLE_SIZE)];
            //z0 = (yz - 1) * y
            weight_t z_0 = (yz-1.0)*y[data_index];
            z0_[data_index] = z_0;
        }

        memset(gradient_, 0, sizeof(weight_t)*weight_dim_);
        #pragma omp parallel for schedule(static)
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            const Dataset::FEATURE_NODE* x = X[data_index];
            while (x->index != -1){
                //Log::Info("%d_%d\n", data_index, X[data_index][index].first);
                OMP_LOOP_EX_BEGIN();
                gradient_[x->index] += x->value * z0_[data_index];
                OMP_LOOP_EX_END();
                x++;
            }
        }
        OMP_THROW_EX();
        for (int dim = 0; dim < weight_dim_; ++dim) {
            gradient_[dim] += alpha* w[dim];
            f_ += alpha*w[dim]*w[dim];
        }
        clock_t end = clock();
        printf("grad time=%f\n",(float)(end-start)*1000/CLOCKS_PER_SEC);
    }



    void Prediction(Dataset::FEATURE_NODE** X, uint32_t num_data, std::vector<label_t>& out_y_pred){
        for (int i = 0; i < num_data; ++i) {
            Dataset::FEATURE_NODE* x = X[i];
            out_y_pred.push_back(predict(x));
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
    weight_t* gradient_;
    weight_t* z0_;
    weight_t f_;
    weight_t* w_;
    weight_t* expTable_;
    weight_t* sigmoidTable_;
};
#endif //FASTLINEAR_BINARY_OBJECTIVE_HPP
