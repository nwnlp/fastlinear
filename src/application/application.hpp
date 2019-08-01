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
    void Init(const std::string& type){
        LoadData();
        Log::Info("File name:%s \n data size:%d feature size %d", dataset_.data_filename_.c_str(),
                dataset_.num_data_,
                dataset_.num_total_features_);
        linear.CreateObjective(type);


    }
    void LoadData(){
        dataset_.LoadFromFile(config_.file_ignore_header, config_.train_file_name);

    }

    void Train(){
        linear.Train(dataset_, config_);
    }
    Config config_;
    Dataset dataset_;
    Linear linear;
};
#endif //FASTLINEAR_APPLICATION_HPP
