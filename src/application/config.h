//
// Created by niw on 2019/8/1.
//

#ifndef FASTLINEAR_CONFIG_H
#define FASTLINEAR_CONFIG_H

struct Config{
public:
    bool file_ignore_header = false;
    const char* train_file_name = "/Users/johnny/Code/LightGBM/examples/binary_classification/binary.train";
    const char* type = "";
    //L2 Norm
    float alpha = 0.1;
    //
    int iterations = 100;

};

#endif //FASTLINEAR_CONFIG_H

