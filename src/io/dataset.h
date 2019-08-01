//
// Created by niw on 2019/7/31.
//

#ifndef FASTLINEAR_DATASET_H
#define FASTLINEAR_DATASET_H


#ifdef WEIGHT_T_USE_DOUBLE
typedef double weight_t;
#define LBFGS_FLOAT = 64
#else
typedef float weight_t;
#define LBFGS_FLOAT = 32
#endif

#ifdef LABEL_T_USE_DOUBLE
typedef double label_t;
#else
typedef float label_t;
#endif

#include <string>
#include <parser.hpp>

class Dataset {
public:

    typedef std::vector<std::pair<int, weight_t>> FEATURE_VALUE;
    typedef std::vector<FEATURE_VALUE> DATA_MAT;
    typedef std::vector<label_t > LABEL_VEC;
    void LoadFromFile(bool ignore_header, const char* filename);

    std::string data_filename_;
    uint32_t num_data_;
    uint32_t num_total_features_;

    int label_idx_;
    DATA_MAT data_;
    LABEL_VEC labels_;
};
#endif //FASTLINEAR_DATASET_H
