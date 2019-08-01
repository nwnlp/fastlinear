//
// Created by niw on 2019/7/31.
//
#include "dataset.h"
#include <text_reader.h>


void Dataset::LoadFromFile(bool ignore_header, const char* filename){

    TextReader<uint32_t> text_reader(filename, ignore_header);
    uint32_t num_global_data = text_reader.ReadAllLines();
    std::vector<std::string> text_data;
    text_data.swap(text_reader.Lines());
    num_data_ = static_cast<uint32_t>(text_data.size());

    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename, ignore_header, 0, label_idx_));
    if (parser == nullptr) {
        Log::Fatal("Could not recognize data format of %s", filename);
    }
    data_filename_ = filename;
    weight_t label;
    FEATURE_VALUE oneline_features;
    for (int i = 0; i < num_data_; ++i) {
        // parse features
        oneline_features.clear();
        parser->ParseOneLine(text_data[i].c_str(), &oneline_features, &label);
        data_.emplace_back(oneline_features);
        labels_.push_back(label);
    }
    num_total_features_ = parser->TotalColumns();

}