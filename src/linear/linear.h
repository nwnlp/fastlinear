//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_LINEAR_H
#define FASTLINEAR_LINEAR_H
#include <memory>
#include <string>
#include <binary_objective.hpp>
#include <dataset.h>
#include <config.h>
#include <lbfgs.h>
#include "tron.h"

class Linear:public function{
public:
    ~Linear(){

    }
    void CreateObjective(const MODEL_TYPE type);
    void Train(Dataset& dataset, Config& config);
    void Predict(Dataset& dataset);
    static lbfgsfloatval_t _evaluate(
            void *instance,
            const lbfgsfloatval_t *x,
            lbfgsfloatval_t *g,
            const int n,
            const lbfgsfloatval_t step
    );

    lbfgsfloatval_t evaluate(
            const lbfgsfloatval_t *x,
            lbfgsfloatval_t *g,
            const int n,
            const lbfgsfloatval_t step
    );

    static int _progress(
            void *instance,
            const lbfgsfloatval_t *x,
            const lbfgsfloatval_t *g,
            const lbfgsfloatval_t fx,
            const lbfgsfloatval_t xnorm,
            const lbfgsfloatval_t gnorm,
            const lbfgsfloatval_t step,
            int n,
            int k,
            int ls
    );


    int progress(
            const lbfgsfloatval_t *x,
            const lbfgsfloatval_t *g,
            const lbfgsfloatval_t fx,
            const lbfgsfloatval_t xnorm,
            const lbfgsfloatval_t gnorm,
            const lbfgsfloatval_t step,
            int n,
            int k,
            int ls
    );

    static void tron_progress(const char *buf);

    //! TRON needs theses interfaces
    double fun(double *w){
        weight_t loss = function_->CalcLoss(dataset_ptr->data_, dataset_ptr->y_, w, config_ptr->alpha);
        printf("func = %lf\n", loss);
        return loss;
    }
    void grad(double *w, double *g){
        function_->CalcGradients(dataset_ptr->data_, dataset_ptr->y_, w, config_ptr->alpha, g);
    }
    void Hv(double *s, double *Hs){
        function_->CalcHv(dataset_ptr->data_, s, Hs);
    };

    int get_nr_variable(void){
        return dataset_ptr->num_total_features_;
    }
    void get_diag_preconditioner(double *M){
        return function_->CalcDiagPreConditioner(dataset_ptr->data_,M);
    }

private:
    void OutputPrediction(const std::string& filename, const std::vector<label_t >y_pred);
    void SaveModel(const std::string& model_file, weight_t* w, uint32_t num_w);
private:
    std::unique_ptr<ObjectiveFunction> function_= nullptr;
    Dataset* dataset_ptr;
    Config* config_ptr;
    std::string model_name;
};
#endif //FASTLINEAR_LINEAR_H
