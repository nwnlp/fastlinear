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
#include <alglib/optimization.h>
using namespace alglib;
class Linear{
public:
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

    static void alglib_function_grad_(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr);
    void alglib_function_grad(const real_1d_array &x, double &func, real_1d_array &grad);
private:
    void PrepareTrainData();
    void OutputPrediction(const std::string& filename, const std::vector<label_t >y_pred);
    void SaveModel(const std::string& model_file, weight_t* w, uint32_t num_w);
private:
    std::unique_ptr<ObjectiveFunction> function_= nullptr;
    Dataset* dataset_ptr;
    Config* config_ptr;
    bool use_0_as_neg_label_ = false;
};
#endif //FASTLINEAR_LINEAR_H
