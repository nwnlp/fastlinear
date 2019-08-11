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
    data_ = new FEATURE_NODE*[num_data_];
    y_ = new weight_t[num_data_];
    labels_.clear();
    std::vector<std::pair<int, weight_t>> oneline_features;
    int max_feature_index = -1;
    for (int i = 0; i < num_data_; ++i) {
        // parse features
        oneline_features.clear();
        parser->ParseOneLine(text_data[i].c_str(), &oneline_features, &label);
        int value_cnt = oneline_features.size();
        data_[i] = new FEATURE_NODE[value_cnt+1];
        for (int j = 0; j < value_cnt; ++j) {
            int feature_index = oneline_features[j].first;
            weight_t feature_value = oneline_features[j].second/255;
            if(feature_min_values_.size() < feature_index+1){
                feature_min_values_.resize(feature_index+1);
                feature_min_values_[feature_index]=std::numeric_limits<weight_t >::infinity();

                feature_max_values_.resize(feature_index+1);
                feature_max_values_[feature_index]=-std::numeric_limits<weight_t >::infinity();
            }
            feature_min_values_[feature_index] = std::min(feature_min_values_[feature_index], feature_value);
            feature_max_values_[feature_index] = std::max(feature_max_values_[feature_index], feature_value);
            max_feature_index = std::max(max_feature_index, feature_index);
            data_[i][j].index = feature_index;
            data_[i][j].value = feature_value;
        }
        data_[i][value_cnt].index = -1;
        if(is_cls_prob){
            label_count[static_cast<int>(label)] += 1;
            labels_.push_back(static_cast<int>(label));
        }
        else{
            //regression
            y_[i] = label;
        }
    }
    num_total_features_ = std::max(parser->TotalColumns()-1, max_feature_index+1);

    if(is_cls_prob){
        if(label_count.size() == 2){
            //!binary classification
            std::map<int, int>::iterator st = label_count.begin();
            int label1 = st->first;
            int label1_cnt = st->second;
            st++;
            int label2 = st->first;
            int label2_cnt = st->second;
            if(label1 > label2){
                Common::swap(label1, label2);
            }
            //!label1 as -1 label2 as 1
            for (int data_index = 0; data_index < labels_.size(); ++data_index) {
                if(labels_[data_index] == label1){
                    y_[data_index] = -1.0;
                }
                else{
                    y_[data_index] = 1.0;
                }
            }
            y2label_[-1.0] = label1;
            y2label_[1.0] = label2;

        }else{
            //!multiple classification
            for (int data_index = 0; data_index < labels_.size(); ++data_index) {
                y_[data_index] = labels_[data_index];


            }
        }
        num_class_ = label_count.size();
    }
}

