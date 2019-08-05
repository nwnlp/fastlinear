//
// Created by johnny on 2019/8/3.
//

#ifndef FASTLINEAR_BINARY_METRICS_HPP
#define FASTLINEAR_BINARY_METRICS_HPP

#include <dataset.h>
class Binary_Metrics{

public:
    void Init(const label_t* y_truth, const label_t* y_pred_score, const uint32_t num_data){
        y_truth_ptr = y_truth;
        y_pred_score_ptr = y_pred_score;
        num_data_ = num_data;
    }
    label_t logloss(){
        label_t loss = 0.0;
        for (int data_index = 0; data_index < num_data_; ++data_index) {
            label_t truth = y_truth_ptr[data_index];
            label_t score = y_pred_score_ptr[data_index];
            if(truth == 1.0){
                loss += std::log(score);
            }else{
                loss += std::log(1-score);
            }
        }
        return -1.0*loss;
    }
    const label_t* y_truth_ptr;
    const label_t* y_pred_score_ptr;
    uint32_t num_data_;

};
#endif //FASTLINEAR_BINARY_METRICS_HPP
