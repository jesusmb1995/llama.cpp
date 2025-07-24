#pragma once

#include <stdint.h>
#include <string>
#include <variant>
#include <vector>

namespace load_input_variant {

struct fname_load_input {
    const std::string &        fname;
    std::vector<std::string> & splits;  // optional, only need if the split does not follow naming scheme
};

struct buffer_load_input {
    const uint8_t * data;
    size_t          size;
};

struct buffer_future_load_input {
    const std::string &        promise_key;
    const std::string &        context;
    std::vector<std::string> & splits;
};

}  // namespace load_input_variant

using load_input_t = std::variant<load_input_variant::fname_load_input, load_input_variant::buffer_load_input,
                                  load_input_variant::buffer_future_load_input>;

namespace load_input_variant {
const char * identifier(load_input_t load_input);

fname_load_input split_name_from_variant(load_input_t & load_input);

bool variant_supports_split_load(load_input_t & load_input);

bool variant_supports_split_load_from_memory(load_input_t & load_input);
}  // namespace load_input_variant

