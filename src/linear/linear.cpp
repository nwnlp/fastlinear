//
// Created by niw on 2019/7/31.
//
#include "linear.h"

void Linear::CreateObjective(const std::string& type){
    if(type == "logistic regression"){
        function_.reset(new BinaryObjective);
    }
}

lbfgsfloatval_t Linear::_evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
)
{
    return reinterpret_cast<Linear*>(instance)->evaluate(x, g, n, step);
}

int Linear::_progress(
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
)
{
    return reinterpret_cast<Linear*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
}

lbfgsfloatval_t Linear::evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
)
{
    function_->CalcGradients(dataset_ptr->data_, dataset_ptr->labels_, x, config_ptr->alpha);
    memcpy(g, function_->gradient(), dataset_ptr->num_total_features_);
    return function_->loss();
}


int Linear::progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
)
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}



void Linear::Train(Dataset& dataset, Config& config){
    dataset_ptr = &dataset;
    config_ptr = &config;
    function_->Init(dataset.num_data_, dataset.num_total_features_);

    int ret = lbfgs(dataset.num_total_features_, function_->weights(), nullptr, _evaluate, _progress, this, NULL);
    /* Report the result. */
    printf("L-BFGS optimization terminated with status code = %d\n", ret);

}