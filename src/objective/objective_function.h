//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_OBJECTIVE_FUNCTION_H
#define FASTLINEAR_OBJECTIVE_FUNCTION_H

#include <dataset.h>
class ObjectiveFunction{
public:
    virtual void CalcGradients(const Dataset::DATA_MAT& X, const Dataset::LABEL_VEC& y, const weight_t* w, float alpha) = 0;
    virtual void Init(uint32_t data_size, uint32_t weight_dim) =0;
    virtual weight_t* gradient() = 0;
    virtual weight_t loss() = 0;
    virtual weight_t* weights() = 0;


};
#endif //FASTLINEAR_OBJECTIVE_FUNCTION_H
