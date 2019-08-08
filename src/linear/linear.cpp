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
    weight_t loss = function_->CalcLoss(dataset_ptr->data_, dataset_ptr->y_, x, config_ptr->alpha);
    function_->CalcGradients(dataset_ptr->data_, dataset_ptr->y_, x, config_ptr->alpha, g);
    return loss;
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

void Linear::tron_progress(const char *buf){
    Log::Info("%s", buf);
}
void Linear::Train(Dataset& dataset, Config& config){

    dataset_ptr = &dataset;
    config_ptr = &config;
    CreateObjective(config_ptr->model_type);
    //dataset_ptr->Normalize(config_ptr->normalize_data_type);

    function_->Init(dataset.num_data_, dataset.num_total_features_);


    //int ret = lbfgs(dataset.num_total_features_, function_->weights(), nullptr, _evaluate, _progress, this, NULL);
    /* Report the result. */
//    for (int i = 0; i < dataset.num_total_features_; ++i) {
//        if(std::fabs(function_->weights()[i])<kZeroThreshold){
//            continue;
//        }
//        printf("%d:%lf ", i,function_->weights()[i]);
//    }

    //printf("L-BFGS optimization terminated with status code = %s\n", lbfgs_strerror(ret));
    float eps = 0.01;
    int neg = dataset_ptr->label_count[-1];
    int pos = dataset_ptr->label_count[1];
    double primal_solver_tol = eps*std::max(std::min(pos,neg), 1)/dataset_ptr->num_data_;
    double eps_cg = 0.1;
    TRON tron_obj(this, primal_solver_tol, eps_cg);
    printf("%lf %lf\n", primal_solver_tol, eps_cg);
    tron_obj.set_print_string(Linear::tron_progress);
    tron_obj.tron(function_->weights());

    SaveModel("model.txt", function_->weights(), dataset.num_total_features_);



}

void Linear::OutputPrediction(const std::string& filename, const std::vector<label_t >y_pred){
    int label1 = dataset_ptr->y2label_[-1.0];
    int label2 = dataset_ptr->y2label_[1.0];
    std::string out = std::string("labels ") + std::to_string(label1)
            + std::string(" ")+std::to_string(label2);
    out+="\n";
    for (int data_index = 0; data_index < y_pred.size(); ++data_index) {
        int truth_label = dataset_ptr->labels_[data_index];
        label_t positive_prob = y_pred[data_index];
        label_t negtive_prob = 1.0-positive_prob;
        int pred_label = negtive_prob>positive_prob?label1:label2;
        out += std::to_string(truth_label) + " "+ std::to_string(pred_label)+" "
                +std::to_string(negtive_prob)+" "
                +std::to_string(positive_prob)+"\n";
    }
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
    metrics.Init(dataset.y_, y_pred.data(), dataset.num_data_);
    printf("log loss:%lf\n",metrics.logloss());
    OutputPrediction("fl_prediction.txt",y_pred);
}