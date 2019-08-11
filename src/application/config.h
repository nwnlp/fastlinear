//
// Created by niw on 2019/8/1.
//

#ifndef FASTLINEAR_CONFIG_H
#define FASTLINEAR_CONFIG_H

enum MODEL_TYPE{
    LOGISTIC_REGRESSION,
    SOFTMAX,
    FM,

};
struct Config{
public:
    bool file_ignore_header = true;
    /*!
     *index of label column start from 0*/
    int label_idx= 0;
    bool fit_intercept = false;
    const char* normalize_data_type = "min_max";
    const char* train_file_name = "/Users/johnny/Downloads/mnist-in-csv/mnist_train.csv";
    MODEL_TYPE model_type = SOFTMAX;
    //L2 Norm
    float alpha = 0.05;
    //
    int iterations = 100;

};

#endif //FASTLINEAR_CONFIG_H

