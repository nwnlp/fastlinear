//
// Created by niw on 2019/8/1.
//

#ifndef FASTLINEAR_CONFIG_H
#define FASTLINEAR_CONFIG_H

enum TASK_TYPE{
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
};
enum MODEL_TYPE{
    LOGISTIC_REGRESSION,
    FM,

};
struct Config{
public:
    bool file_ignore_header = false;
    /*!
     *index of label column start from 0*/
    int label_idx= 0;
    bool fit_intercept = false;
    const char* normalize_data_type = "min_max";
    const char* train_file_name = "E:\\kaggle\\liblinear-master\\news20.binary";
    MODEL_TYPE model_type = LOGISTIC_REGRESSION;
    TASK_TYPE take_type = BINARY_CLASSIFICATION;
    //L2 Norm
    float alpha = 0.01;
    //
    int iterations = 100;

};

#endif //FASTLINEAR_CONFIG_H

