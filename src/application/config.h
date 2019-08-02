//
// Created by niw on 2019/8/1.
//

#ifndef FASTLINEAR_CONFIG_H
#define FASTLINEAR_CONFIG_H

struct Config{
public:
    bool file_ignore_header = true;
    /*!
     *index of label column start from 0*/
    int label_idx= 30;
    bool fit_intercept = true;

    const char* train_file_name = "E:\\kaggle\\ctr-rank\\tree\\fasttree\\train.csv";
    const char* type = "";
    //L2 Norm
    float alpha = 0.0;
    //
    int iterations = 100;

};

#endif //FASTLINEAR_CONFIG_H

