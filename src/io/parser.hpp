/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_PARSER_HPP_
#define LIGHTGBM_IO_PARSER_HPP_

#include <common.h>
#include <log.h>
#include <file_io.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <dataset.h>
/*! \brief Type of data size, it is better to use signed type*/

const weight_t kMinScore = -std::numeric_limits<weight_t >::infinity();

const weight_t kEpsilon = 1e-15f;

const double kZeroThreshold = 1e-35f;


typedef int32_t comm_size_t;

/*! \brief Interface for Parser */
class Parser {
public:
    /*! \brief virtual destructor */
    virtual ~Parser() {}

    /*!
    * \brief Parse one line with label
    * \param str One line record, string format, should end with '\0'
    * \param out_features Output columns, store in (column_idx, values)
    * \param out_label Label will store to this if exists
    */
    virtual void ParseOneLine(const char* str,
                              std::vector<std::pair<int, weight_t >>* out_features, label_t * out_label) const = 0;

    virtual int TotalColumns() const = 0;

    /*!
    * \brief Create a object of parser, will auto choose the format depend on file
    * \param filename One Filename of data
    * \param num_features Pass num_features of this data file if you know, <=0 means don't know
    * \param label_idx index of label column
    * \return Object of parser
    */
    static Parser* CreateParser(const char* filename, bool header, int num_features, int label_idx);
};

class CSVParser: public Parser {
 public:
  explicit CSVParser(int label_idx, int total_columns)
    :label_idx_(label_idx), total_columns_(total_columns) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, weight_t >>* out_features, label_t* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int bias = 0;
    *out_label = 0.0f;
    while (*str != '\0') {
      str = Common::Atof(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        bias = -1;
      } else if (std::fabs(val) > kZeroThreshold && !std::isnan(val)) { //we ignore zero and nan value
        out_features->emplace_back(idx + bias, val);
      }
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as CSV");
      }
    }
  }

  inline int TotalColumns() const override {
    return total_columns_;
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
};

class TSVParser: public Parser {
 public:
  explicit TSVParser(int label_idx, int total_columns)
    :label_idx_(label_idx), total_columns_(total_columns) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, weight_t >>* out_features, label_t * out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int bias = 0;
    while (*str != '\0') {
      str = Common::Atof(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        bias = -1;
      } else if (std::fabs(val) > kZeroThreshold && !std::isnan(val)) {//we ignore zero and nan value
        out_features->emplace_back(idx + bias, val);
      }
      ++idx;
      if (*str == '\t') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as TSV");
      }
    }
  }

  inline int TotalColumns() const override {
    return total_columns_;
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
};

class LibSVMParser: public Parser {
 public:
  explicit LibSVMParser(int label_idx)
    :label_idx_(label_idx) {
    if (label_idx > 0) {
      Log::Fatal("Label should be the first column in a LibSVM file");
    }
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, weight_t >>* out_features, label_t * out_label) const override {
    int idx = 0;
    double val = 0.0f;
    if (label_idx_ == 0) {
      str = Common::Atof(str, &val);
      *out_label = val;
      str = Common::SkipSpaceAndTab(str);
    }
    while (*str != '\0') {
      str = Common::Atoi(str, &idx);
      str = Common::SkipSpaceAndTab(str);
      if (*str == ':') {
        ++str;
        str = Common::Atof(str, &val);
        out_features->emplace_back(idx, val);
      } else {
        Log::Fatal("Input format error when parsing as LibSVM");
      }
      str = Common::SkipSpaceAndTab(str);
    }
  }

  inline int TotalColumns() const override {
    return -1;
  }

 private:
  int label_idx_ = 0;
};

#endif   // LightGBM_IO_PARSER_HPP_
