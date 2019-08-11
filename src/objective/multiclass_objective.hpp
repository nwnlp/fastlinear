//
// Created by johnny on 2019/8/10.
//

#ifndef FASTLINEAR_MULTICLASS_OBJECTIVE_HPP
#define FASTLINEAR_MULTICLASS_OBJECTIVE_HPP

#include "objective_function.h"
class SoftMaxObjective:public ObjectiveFunction{
public:
    ~SoftMaxObjective(){
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            delete[] softmax_[data_index];
        }
        delete[] softmax_;
        delete[] w_;
    }
    void Init(uint32_t data_size, uint32_t weight_dim, int num_class){
        weight_dim_ = weight_dim;
        data_size_ = data_size;
        num_class_ = num_class;
        softmax_ = new weight_t*[data_size];
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            softmax_[data_index] = new weight_t[num_class];
        }
        w_= new weight_t[weight_dim*num_class];
        memset(w_, 0, sizeof(weight_t)*weight_dim*num_class);

    }


    inline weight_t sparse_dot(const Dataset::FEATURE_NODE* a, const weight_t* b){
        weight_t c = 0.0;
        while (a->index != -1) {
            c += b[a->index] * a->value;
            a++;
        }
        return c;
    }

    weight_t CalcLoss(Dataset::FEATURE_NODE** X, weight_t* y, const weight_t* w, float alpha){
        weight_t loss = 0.0;
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            int true_cls = static_cast<int>(y[data_index]);
            std::vector<weight_t > wTx;
            weight_t wtxmax = -std::numeric_limits<weight_t >::infinity();
            for (int c = 0; c < num_class_; ++c) {
                //!WTX
                weight_t wtx = sparse_dot(X[data_index], &w[c * weight_dim_]);
                wTx.push_back(wtx);
                wtxmax = std::max(wtxmax, wtx);
            }
            std::vector<weight_t > exp_wTx;
            weight_t exp_sum = 0.0;
            for (int c = 0; c < num_class_; ++c) {
                weight_t exp_wtx = std::exp(wTx[c]-wtxmax);
                exp_wTx.push_back(exp_wtx);
                exp_sum += exp_wtx;
            }
            for (int c = 0; c < num_class_; ++c) {
                weight_t softmax = exp_wTx[c]/exp_sum;
                softmax_[data_index][c] = softmax;
                if(c == true_cls){
                    loss += (- std::log(softmax));
                }
            }

        }
        for (int dim = 0; dim < weight_dim_*num_class_; ++dim) {
            loss += alpha*w[dim]*w[dim];
        }
        printf("loss=%lf\n", loss);
        return loss;


    }

    void CalcGradients(Dataset::FEATURE_NODE** X, weight_t* y, const weight_t* w, float alpha, weight_t* g){
        memset(g, 0, sizeof(weight_t)*weight_dim_*num_class_);
        for (int data_index = 0; data_index < data_size_; ++data_index) {
            int true_cls = static_cast<int>(y[data_index]);
            Dataset::FEATURE_NODE *x = X[data_index];
            while(x->index != -1){
                //printf("%d, %lf\n", x->index, x->value);
                //printf("grad1=%lf grad2=%lf softmax=%lf true_cls=%d, true_cls_weight_start=%d\n",
                //        grad1, grad2, softmax_[data_index], true_cls, true_cls_weight_start);
                for (int c = 0; c < num_class_; ++c) {
                    uint32_t start = c*weight_dim_;
                    if(c == true_cls){
                        g[start+x->index] += (x->value*(softmax_[data_index][c]-1.0));
                    }
                    else{
                        g[start+x->index] += (x->value*(softmax_[data_index][c]));
                    }
                }
                x++;

            }

        }
        for (int dim = 0; dim < weight_dim_*num_class_; ++dim) {
            g[dim] += 2*alpha* w[dim];
        }
        for (int i = 0; i < 10; ++i) {
            printf("%lf ", g[i]);
        }
        printf("\n");
    }

    void Predict(Dataset::FEATURE_NODE** X, uint32_t num_data, std::vector<int>& out_y_pred){
        for (int i = 0; i < num_data; ++i) {
            Dataset::FEATURE_NODE* x = X[i];
            out_y_pred.push_back(predict(x));
        }

    }
    weight_t* weights(){
        return w_;
    }


private:
    int predict(Dataset::FEATURE_NODE*x){
        weight_t wx=0.0;
        std::vector<weight_t > wTx;
        weight_t wtxmax = -std::numeric_limits<weight_t >::infinity();
        for (int c = 0; c < num_class_; ++c) {
            //!WTX
            weight_t wtx = sparse_dot(x, &w_[c * weight_dim_]);
            wTx.push_back(wtx);
            wtxmax = std::max(wtxmax, wtx);
        }

        weight_t exp_sum = 0.0;
        int max_prob_c = 0;
        weight_t exp_wtx_max = -std::numeric_limits<weight_t >::infinity();
        for (int c = 0; c < num_class_; ++c) {
            weight_t exp_wtx = std::exp(wTx[c]-wtxmax);
            if(exp_wtx > exp_wtx_max){
                exp_wtx_max = exp_wtx;
                max_prob_c = c;
            }
            exp_sum += exp_wtx;
        }
        //max prob
        exp_wtx_max/exp_sum;
        return max_prob_c;

    }
private:
    uint32_t weight_dim_;
    uint32_t data_size_;
    int num_class_;
    weight_t* w_;
    weight_t** softmax_;

};
#endif //FASTLINEAR_MULTICLASS_OBJECTIVE_HPP
