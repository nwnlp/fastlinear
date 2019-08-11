//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_OBJECTIVE_FUNCTION_H
#define FASTLINEAR_OBJECTIVE_FUNCTION_H

#include <dataset.h>
class ObjectiveFunction{
public:
    virtual void CalcGradients(Dataset::FEATURE_NODE** X, weight_t* y, const weight_t* w, float alpha, weight_t* g) = 0;
    virtual weight_t CalcLoss(Dataset::FEATURE_NODE** X, weight_t* y, const weight_t* w, float alpha) = 0;
    virtual void CalcHv(Dataset::FEATURE_NODE** X, weight_t* s, weight_t* Hs) {};
    virtual void CalcDiagPreConditioner(Dataset::FEATURE_NODE** X, weight_t* M) {};
    virtual void Init(uint32_t data_size, uint32_t weight_dim, int num_class = 2) =0;
    virtual weight_t* weights() = 0;

    virtual void PredictScore(Dataset::FEATURE_NODE** X, uint32_t num_data, std::vector<label_t>& out_y_pred){}
    virtual void Predict(Dataset::FEATURE_NODE** X, uint32_t num_data, std::vector<int>& out_y_pred){}
};
#endif //FASTLINEAR_OBJECTIVE_FUNCTION_H
