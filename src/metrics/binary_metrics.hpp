//
// Created by johnny on 2019/8/3.
//

#ifndef FASTLINEAR_BINARY_METRICS_HPP
#define FASTLINEAR_BINARY_METRICS_HPP

#include <dataset.h>
class Binary_Metrics{

public:
    void Init(const Dataset::LABEL_VEC& y_truth, const Dataset::LABEL_VEC& y_pred_score){
        y_truth_ptr = &y_truth;
        y_pred_score_ptr = &y_pred_score;
    }
    label_t logloss(){
        label_t loss = 0.0;
        for (int data_index = 0; data_index < y_truth_ptr->size(); ++data_index) {
            label_t truth = (*y_truth_ptr)[data_index];
            label_t score = (*y_pred_score_ptr)[data_index];
            if(truth == 1.0){
                loss += std::log(score);
            }else{
                loss += std::log(1-score);
            }
        }
        return -1.0*loss;
    }
    const Dataset::LABEL_VEC* y_truth_ptr;
    const Dataset::LABEL_VEC* y_pred_score_ptr;

};
#endif //FASTLINEAR_BINARY_METRICS_HPP
