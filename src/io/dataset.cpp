//
// Created by niw on 2019/7/31.
//
#include "dataset.h"
#include <text_reader.h>


void Dataset::LoadFromFile(bool ignore_header, const char* filename, int label_idx){

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
    for (int i = 0; i < num_data_; ++i) {
        // parse features
        oneline_features.clear();
        parser->ParseOneLine(text_data[i].c_str(), &oneline_features, &label);
        /*for (int j = 0; j < oneline_features.size(); ++j) {
            printf("%d,%f\n", oneline_features[j].first, oneline_features[j].second);
            assert(oneline_features[j].first>=0);
        }*/
        for (int j = 0; j < oneline_features.size(); ++j) {
            int feature_index = oneline_features[j].first;
            weight_t feature_value = oneline_features[j].second;
            if(feature_min_values_.size() < feature_index+1){
                
            }
        }
        data_.emplace_back(oneline_features);
        labels_.push_back(label);
    }
    num_total_features_ = parser->TotalColumns()-1;

}

void Dataset::Normalize(const std::string& type){

}

void Dataset::AddBiasTag(){
    for (int data_index = 0; data_index < num_data_; ++data_index) {
        data_[data_index].push_back(std::make_pair<int, weight_t>(num_total_features_, 1.0));
    }
    num_total_features_+=1;
}