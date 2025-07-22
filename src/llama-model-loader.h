#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <variant>

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED = 1;
    static const int TENSOR_DUPLICATED   = 2;

    int n_kv      = 0;
    mutable int n_tensors = -1;
    int max_n_tensors = -1;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool check_tensors;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context_ptr meta;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    struct fname_load_input {
        const std::string& fname;
        std::vector<std::string>& splits; // optional, only need if the split does not follow naming scheme
    };

    struct buffer_load_input {
        const uint8_t * data;
        size_t size;
    };

    struct buffer_future_load_input {
        const std::string& promise_key;
        const std::string& context;
        std::vector<std::string>& splits;
    };

    using load_input_t = std::variant<fname_load_input, buffer_load_input, buffer_future_load_input>;

    llama_model_loader(
        load_input_t load_input,
        bool use_mmap,
        bool check_tensors,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const;

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::initializer_list<int64_t> & ne, int flags = 0);

    struct ggml_tensor * create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base, const std::string & name, const std::initializer_list<int64_t> & ne, size_t offset, bool required = true);

    void done_getting_tensors() const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // for backwards compatibility, does not support ggml-backend
    void load_data_for(struct ggml_tensor * cur) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;

    struct gguf_file_load {
        struct gguf_init_params params;
        gguf_context_ptr        meta;
        llama_file *            file = nullptr;

        gguf_file_load(struct ggml_context ** ctx, llama_model_loader::load_input_t load_input);

        static const char * identifier(llama_model_loader::load_input_t load_input);

        static gguf_file_load load_split_gguf(struct ggml_context ** ctx, const char * fname_split,
                                              llama_model_loader::load_input_t & load_input,
                                              std::vector<std::string> &         splits);

        static llama_model_loader::fname_load_input split_name_from_variant(
            llama_model_loader::load_input_t & load_input);

        static bool variant_supports_split_load(llama_model_loader::load_input_t & load_input);

        static bool variant_supports_split_load_from_memory(llama_model_loader::load_input_t & load_input);
    };

    /// @brief Stores necessary information to load a weights from a split file. But does not
    /// immediatelly trigger the loading and buffer storage. Instead, `load` function has to be
    /// called. Useful for progressive load where file weights are only read once both tensor buffer
    /// and ready for the upload.
    struct SplitWeightDelayedLoad : public llama_file {
        // TODO: un-mutable approach
        mutable llama_model_loader::load_input_t load_input;
        llama_model_loader &                     loader;
        llama_model_loader::fname_load_input     base_split;
        uint16_t                                 idx;
        std::string                              kv_split_no;
        mutable bool                             loaded = false;
        mutable gguf_file_load *                 split_gguf;  // TODO smart-pointer or smarter stack allocation

        SplitWeightDelayedLoad(llama_model_loader::load_input_t load_input, llama_model_loader & loader,
                               llama_model_loader::fname_load_input base_split, uint16_t idx, std::string kv_split_no);
        SplitWeightDelayedLoad(SplitWeightDelayedLoad&& other) noexcept;

        ~SplitWeightDelayedLoad() override;
        size_t   tell() const override;
        size_t   size() const override;
        int      file_id() const override;
        void     seek(size_t offset, int whence) const override;
        void     read_raw(void * ptr, size_t len) const override;
        uint32_t read_u32() const override;
        void     write_raw(const void * ptr, size_t len) const override;
        void     write_u32(uint32_t val) const override;
        void     load() const;
    };

    std::size_t delayed_loaded = 0;
    std::vector<SplitWeightDelayedLoad*> delayed_files;
};
