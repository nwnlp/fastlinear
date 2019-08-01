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
class Linear{
public:
    void CreateObjective(const std::string& type);
    void Train(Dataset& dataset, Config& config);
private:
    std::unique_ptr<ObjectiveFunction> function_= nullptr;
};
#endif //FASTLINEAR_LINEAR_H
