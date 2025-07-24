#pragma once

#include <cstdint>

#include "ggml-cpp.h"
#include "llama-mmap.h"
#include "llama-model-load-input.h"

struct llama_model_loader;

/// @brief Immediately loads and stores relevant data in the struct fields.
struct gguf_file_load {
    struct gguf_init_params     params;
    gguf_context_ptr            meta;
    std::unique_ptr<llama_file> file = nullptr;

    gguf_file_load(struct ggml_context ** ctx, load_input_t load_input);
};

/// @brief Stores relevant information to be able to loads a `.gguf` split file when load method is called.
struct SplitLoad {
    load_input_t                         load_input;
    load_input_variant::fname_load_input base_split;
    uint16_t                             idx;
    std::string                          kv_split_no;
    bool                                 loaded = false;

    SplitLoad(load_input_t load_input, load_input_variant::fname_load_input base_split, uint16_t idx,
              std::string kv_split_no);

    static gguf_file_load load_split_gguf(struct ggml_context ** ctx, const char * fname_split,
                                          load_input_t & load_input, std::vector<std::string> & splits);

    struct ggml_context * load(struct llama_model_loader & ml);
};

/// @brief Handles incremental load of tensor and split-files.
/// @note First split-file will be immediately load at construction, the remainder of split-files are
/// incrementally load on-demand by calling `loadTensorMetadata`
struct SplitsTensorLoad {

    SplitsTensorLoad(struct ggml_context* ctx, struct llama_model_loader & ml, gguf_file_load& base_split);

    void addSplit(SplitLoad splitLoad);

    /// @brief Incrementally loads file splits until the tensor metadata is found.
    /// Also increments loaded tensor count so that `allTensorsAreLoaded` returns true
    /// when all tensors in a file-split have been requested.
    /// @returns Split idx where the tensor was found
    /// @throw runtime_error if tensor was not found
    uint16_t loadTensorMetadata(struct llama_model_loader & ml, const char * tensor_name,
                                ggml_tensor ** out_tensor_metadata);

    /// @returns True if all tensors of a split have been loaded.
    bool allTensorsAreLoaded(uint16_t split_idx) const;

    /// @bried Release file memory for a split.
    static void releaseSplit(struct llama_model_loader & ml, uint16_t split_idx);

    void printCurrentlyKnownTensors() const;

    uint16_t getSplitIdxForTensor(const char * tensor_name) const;

    std::size_t getSplitDataSize(uint16_t split_idx) const;

private:
    void loadSplit(struct llama_model_loader & ml, uint16_t idx);
    void processSplit(const struct ggml_context * ctx, struct llama_model_loader & ml, uint16_t idx);

    std::map<std::string, int> tensor_to_split;
    std::map<std::string, bool> tensor_to_loaded;
    std::map<int, int>         split_to_tensor_count; // TODO consolidate maps into single one
    std::map<int, std::size_t> split_to_size_data;

    std::size_t            delayed_loaded = 0;
    std::vector<SplitLoad> delayed_files;
    std::map<int, int>     split_to_loaded_tensor_count; // TODO rename to requested
};
