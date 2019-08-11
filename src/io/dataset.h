//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_DATASET_H
#define FASTLINEAR_DATASET_H



#include <string>
#include <parser.hpp>
#include <map>
class Dataset {
public:
    struct FEATURE_NODE{
        int index;
        weight_t value;
    };
    ~Dataset(){

        if(data_){
            for (int i = 0; i < num_data_; ++i) {
                delete[] data_[i];
            }
            delete[] data_;
        }
        if(y_){
            delete[] y_;
        }
    }
    void LoadFromFile(bool ignore_header, const char* filename, int label_idx,bool is_cls_prob);
    void PrintInfo(){
        Log::Info("Load data from %s \n data size:%d feature size:%d", data_filename_.c_str(),
                num_data_, num_total_features_);
        if(!label_count.empty()){
            std::map<int,int>::iterator st = label_count.begin();
            while (st != label_count.end()){
                Log::Info("label:%d count:%d", st->first, st->second);
                st++;

            }

        }
    }
    std::string data_filename_;
    uint32_t num_data_;
    uint32_t num_total_features_;
    int num_class_;
    FEATURE_NODE** data_ = nullptr;
    weight_t* y_ = nullptr;
    std::vector<int> labels_;
    std::map<int, int> label_count;
    std::map<label_t ,char> y2label_;
    std::vector<weight_t> feature_max_values_;
    std::vector<weight_t> feature_min_values_;
};
#endif //FASTLINEAR_DATASET_H
