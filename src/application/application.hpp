//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_APPLICATION_HPP
#define FASTLINEAR_APPLICATION_HPP
#include <dataset.h>
#include <linear.h>
#include <config.h>
class Application{
public:
    void Init(){
        LoadData();
        dataset_.PrintInfo();


    }
    void LoadData(){
        bool is_cls_prob = !(config_.take_type == REGRESSION);
        dataset_.LoadFromFile(config_.file_ignore_header, config_.train_file_name, config_.label_idx,is_cls_prob);

    }

    void Train(){
        linear.Train(dataset_, config_);
    }

    void Predict(){
        linear.Predict(dataset_);
    }
    Config config_;
    Dataset dataset_;
    Linear linear;
};
#endif //FASTLINEAR_APPLICATION_HPP
