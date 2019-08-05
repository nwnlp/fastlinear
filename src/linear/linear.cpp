//
// Created by niw on 2019/7/31.
//
#include "linear.h"
#include <fstream>
#include <binary_metrics.hpp>
void Linear::CreateObjective(const MODEL_TYPE type){
    if(type ==  LOGISTIC_REGRESSION){
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
    return reinterpret_cast<Linear*>(ptr)->alglib_function_grad( x, func, grad);

}

void Linear::PrepareTrainData(){
    if(config_ptr->take_type == BINARY_CLASSIFICATION){
        //EncodeLabel(dataset_ptr->labels_);
        uint32_t label_cnt = dataset_ptr->label_count.size();
        if(label_cnt != 2){
            Log::Fatal("Invalid Data for binary classification, count of distinct label:%d", label_cnt);
        }
        if(dataset_ptr->label_count.find(-1) != dataset_ptr->label_count.end()
        and dataset_ptr->label_count.find(1) != dataset_ptr->label_count.end()){
            //raw data's labels is -1 and 1
        }else if(dataset_ptr->label_count.find(0.0) != dataset_ptr->label_count.end()
                 and dataset_ptr->label_count.find(1) != dataset_ptr->label_count.end()){
            //row data's labels is 0 and 1
            for (int data_index = 0; data_index < dataset_ptr->num_data_; ++data_index) {
                if(std::fabs(dataset_ptr->labels_[data_index]) < kZeroThreshold){
                    dataset_ptr->labels_[data_index] = -1.0;
                }
            }
            use_0_as_neg_label_ = true;
        }
        else{
            Log::Fatal("Invalid Data for binary classification, use 0/1 or -1/1 as labels");
        }

    }
    //normalize

    if(config_ptr->fit_intercept){
       // dataset_ptr->AddBiasTag();
    }


}

void Linear::Train(Dataset& dataset, Config& config){

    dataset_ptr = &dataset;
    config_ptr = &config;
    CreateObjective(config_ptr->model_type);
    PrepareTrainData();
    //dataset_ptr->Normalize(config_ptr->normalize_data_type);

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
//    for (int i = 0; i < dataset.num_total_features_; ++i) {
//        if(std::fabs(function_->weights()[i])<kZeroThreshold){
//            continue;
//        }
//        printf("%d:%lf ", i,function_->weights()[i]);
//    }
    printf("L-BFGS optimization terminated with status code = %s\n", lbfgs_strerror(ret));
    SaveModel("model.txt", function_->weights(), dataset.num_total_features_);



}

void Linear::OutputPrediction(const std::string& filename, const std::vector<label_t >y_pred){

    std::string out = std::string("labels 1 ") + (use_0_as_neg_label_?"0":"-1");
    out+="\n";
    for (int data_index = 0; data_index < y_pred.size(); ++data_index) {
        label_t label = dataset_ptr->labels_[data_index];
        label_t positive_prob = y_pred[data_index];
        label_t negtive_prob = 1.0-positive_prob;
        out += std::to_string(label) + " "+std::to_string(positive_prob)+" "+std::to_string(negtive_prob)+"\n";
    }
    out[out.size()-1]=0;
    std::ofstream fout(filename);
    fout<<out;
    fout.close();


}

void Linear::SaveModel(const std::string& model_file, weight_t* w, uint32_t num_w){
    std::string out = "w:\n";
    for (int i = 0; i < num_w; ++i) {
        out += std::to_string(w[i])+"\n";
    }
    out[out.size()-1]=0;
    std::ofstream fout(model_file);
    fout<<out;
    fout.close();
}

void Linear::Predict(Dataset& dataset){
    std::vector<label_t >y_pred;
    function_->Prediction(dataset.data_, dataset.num_data_, y_pred);
    Binary_Metrics metrics;
    metrics.Init(dataset.labels_, y_pred.data(), dataset.num_data_);
    printf("log loss:%lf\n",metrics.logloss());
    OutputPrediction("fl_prediction.txt",y_pred);
}