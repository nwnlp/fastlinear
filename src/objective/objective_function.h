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
    virtual void CalcHv(Dataset::FEATURE_NODE** X, weight_t* s, weight_t* Hs) = 0;
    virtual void CalcDiagPreConditioner(Dataset::FEATURE_NODE** X, weight_t* M) = 0;
    virtual void Init(uint32_t data_size, uint32_t weight_dim) =0;
    virtual weight_t* weights() = 0;
    virtual void Prediction(Dataset::FEATURE_NODE** X, uint32_t num_data, std::vector<weight_t>& out_y_pred) = 0;

protected:
    virtual label_t predict(Dataset::FEATURE_NODE* feature_values) = 0;
};
#endif //FASTLINEAR_OBJECTIVE_FUNCTION_H
