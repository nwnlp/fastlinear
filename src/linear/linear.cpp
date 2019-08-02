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

void Linear::alglib_function_grad(const real_1d_array &x, double &func, real_1d_array &grad)
{
//    func = 100*pow(x[0]+3,4) + pow(x[1]-3,4);
//    grad[0] = 400*pow(x[0]+3,3);
//    grad[1] = 4*pow(x[1]-3,3);
//    printf("fx:%lf\n",func);
    function_->CalcGradients(dataset_ptr->data_, dataset_ptr->labels_, x.getcontent(), config_ptr->alpha);
    memcpy(grad.getcontent(), function_->gradient(), dataset_ptr->num_total_features_);
    func = function_->loss();
    printf("fx:%lf\n",func);
}


void Linear::alglib_function_grad_(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    reinterpret_cast<Linear*>(ptr)->alglib_function_grad( x, func, grad);
    return;
}

void Linear::Train(Dataset& dataset, Config& config){
    dataset_ptr = &dataset;
    config_ptr = &config;
    if(config.fit_intercept){
        dataset_ptr->AddBiasTag();
    }
    function_->Init(dataset.num_data_, dataset.num_total_features_);
//    real_1d_array x;
//    x.setcontent(dataset.num_total_features_, function_->weights());
//    real_1d_array s = "[1,1]";
//    double epsg = 0;
//    double epsf = 0;
//    double epsx = 0.0000000001;
//    ae_int_t maxits = 0;
//    minlbfgsstate state;
//    minlbfgscreate(1, x, state);
//    minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//    //minlbfgssetscale(state, s);
//    minlbfgsoptguardsmoothness(state);
//    minlbfgsoptguardgradient(state, 0.001);
//    minlbfgsreport rep;
//    minlbfgsoptimize(state, alglib_function_grad_, nullptr, this);
//    minlbfgsresults(state, x, rep);
//    printf("%s\n", x.tostring(2).c_str()); // EXPECTED: [-3,3]
//    optguardreport ogrep;
//    minlbfgsoptguardresults(state, ogrep);
//    printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
//    printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
//    printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false

    int ret = lbfgs(dataset.num_total_features_, function_->weights(), nullptr, _evaluate, _progress, this, NULL);
    /* Report the result. */
    for (int i = 0; i < dataset.num_total_features_; ++i) {
        printf("%lf ", function_->weights()[i]);
    }
    printf("L-BFGS optimization terminated with status code = %s\n", lbfgs_strerror(ret));


}