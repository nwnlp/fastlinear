//
// Created by niw on 2019/7/31.
//
#include "dataset.h"
#include <text_reader.h>
#include <set>

void Dataset::LoadFromFile(bool ignore_header, const char* filename, int label_idx, bool is_cls_prob){

    TextReader<uint32_t> text_reader(filename, ignore_header);
    uint32_t num_global_data = text_reader.ReadAllLines();
    std::vector<std::string> text_data;
    text_data.swap(text_reader.Lines());
    num_data_ = static_cast<uint32_t>(text_data.size());

    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename, ignore_header, 0, label_idx));
    if (parser == nullptr) {
        Log::Fatal("Could not recognize data format of %s", filename);
    }
    data_filename_ = filename;
    label_t label;
    FEATURE_VALUE oneline_features;
    int max_feature_index = -1;
    for (int i = 0; i < num_data_; ++i) {
        // parse features
        oneline_features.clear();
        parser->ParseOneLine(text_data[i].c_str(), &oneline_features, &label);
        for (int j = 0; j < oneline_features.size(); ++j) {
            int feature_index = oneline_features[j].first;
            weight_t feature_value = oneline_features[j].second;
            if(feature_min_values_.size() < feature_index+1){
                feature_min_values_.resize(feature_index+1);
                feature_min_values_[feature_index]=std::numeric_limits<weight_t >::infinity();

                feature_max_values_.resize(feature_index+1);
                feature_max_values_[feature_index]=-std::numeric_limits<weight_t >::infinity();
            }
            feature_min_values_[feature_index] = std::min(feature_min_values_[feature_index], feature_value);
            feature_max_values_[feature_index] = std::max(feature_max_values_[feature_index], feature_value);
            max_feature_index = std::max(max_feature_index, feature_index);
        }
        data_.emplace_back(oneline_features);
        labels_.push_back(label);
        if(is_cls_prob)
            label_count[label] += 1;
    }
    num_total_features_ = std::max(parser->TotalColumns()-1, max_feature_index+1);

}

void Dataset::Normalize(const std::string& type){
    std::vector<weight_t> feature_max_min_values_diff;
    feature_max_min_values_diff.resize(num_total_features_);
    std::set<int> ignore_features;
    for (int k = 0; k < num_total_features_; ++k) {
        feature_max_min_values_diff[k] = feature_max_values_[k]-feature_min_values_[k];
    }
    for (int i = 0; i < num_data_; ++i) {
        FEATURE_VALUE feature_values;
        for (int j = 0; j < data_[i].size(); ++j) {

            int feature_index = data_[i][j].first;
            int feature_value = data_[i][j].second;
            weight_t norm_value = feature_value;
            //if(!Common::EqualTpZero(feature_max_min_values_diff[feature_index])){
            if(std::fabs(feature_max_min_values_diff[feature_index]) > kZeroThreshold){
                norm_value = (feature_value-feature_min_values_[feature_index])
                                      /feature_max_min_values_diff[feature_index];
            }

            feature_values.emplace_back(std::make_pair(feature_index, norm_value));
        }
        data_[i] = feature_values;
    }
    num_total_features_ = num_total_features_-ignore_features.size();
}

void Dataset::AddBiasTag(){
    for (int data_index = 0; data_index < num_data_; ++data_index) {
        data_[data_index].push_back(std::make_pair<int, weight_t>(num_total_features_, 1.0));
    }
    num_total_features_+=1;
}