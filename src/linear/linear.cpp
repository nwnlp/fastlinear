//
// Created by niw on 2019/7/31.
//
#include "linear.h"
#include <lbfgs.h>
void Linear::CreateObjective(const std::string& type){
    if(type == "logistic regression"){
        function_.reset(new BinaryObjective);
    }
}

void Linear::Train(Dataset& dataset, Config& config){
    weight_t * gradient = nullptr;
    function_->Init(dataset.num_data_, dataset.num_total_features_);
    //function_->GetGradients(dataset.data_, dataset.labels_, config.alpha, &gradient);

    int ret = lbfgs(dataset.num_total_features_, gradient, &fx, _evaluate, _progress, this, NULL);

}