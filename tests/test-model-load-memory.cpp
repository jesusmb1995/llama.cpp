#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "get-model.h"
#include "llama-cpp.h"

namespace {
std::vector<std::uint8_t> load_file_into_memory(const char * const model_path) {
    std::ifstream file_stream(model_path, std::ios::binary | std::ios::ate);
    if (!file_stream) {
        fprintf(stderr, "Failed to open file for reading into buffer\n");
        exit(EXIT_FAILURE);
    }

    const size_t file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    static_assert(sizeof(std::uint8_t) == sizeof(char), "uint8_t must be same size as char");
    std::vector<std::uint8_t> buffer(file_size);
    if (!file_stream.read((char*) buffer.data(), file_size)) {
        fprintf(stderr, "Failed to read entire file into buffer\n");
        exit(EXIT_FAILURE);
    }

    return buffer;
}
}  // namespace

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);

    // Manually load into a memory buffer first
    std::vector<std::uint8_t> buffer = load_file_into_memory(model_path);

    llama_backend_init();
    auto params              = llama_model_params{};
    params.use_mmap          = false;
    params.progress_callback = [](float progress, void * ctx) {
        (void) ctx;
        fprintf(stderr, "%.2f%% ", progress * 100.0f);
        // true means: Don't cancel the load
        return true;
    };

    // Test that it can load directly from a buffer
    printf("Loading model from buffer of size %zu bytes\n", buffer.size());
    auto * model = llama_model_load_from_buffer(std::move(buffer), params);

    // Add newline after progress output
    fprintf(stderr, "\n");

    if (model == nullptr) {
        fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Model loaded successfully\n");
    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}
