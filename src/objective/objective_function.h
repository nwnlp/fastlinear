//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_OBJECTIVE_FUNCTION_H
#define FASTLINEAR_OBJECTIVE_FUNCTION_H

#include <dataset.h>
class ObjectiveFunction{
public:
    virtual void GetGradients(Dataset::DATA_MAT& X, const Dataset::LABEL_VEC& y, float alpha , weight_t** gradients) const = 0;
    virtual void Init(uint32_t data_size, uint32_t weight_dim) =0;


};
#endif //FASTLINEAR_OBJECTIVE_FUNCTION_H
