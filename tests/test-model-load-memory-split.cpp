#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <thread>
#include <sstream>

#include "get-model.h"
#include "llama.h"

namespace {
std::vector<std::uint8_t> load_file_into_memory(const char * const model_path) {
    std::ifstream file_stream(model_path, std::ios::binary | std::ios::ate);
    if (!file_stream) {
        fprintf(stderr, "Failed to open file %s for reading into buffer\n", model_path);
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

struct file_entry {
    std::string path;
    std::vector<std::uint8_t> buffer;
};

std::vector<file_entry> load_files_into_memory(const char * const model_path) {
    std::vector<file_entry> files;

    // Extract pattern from first file path
    std::string path(model_path);

    // Split by '-'
    std::vector<std::string> parts;
    std::stringstream ss(path);
    std::string item;
    while (std::getline(ss, item, '-')) {
        parts.push_back(item);
    }

    // Split the last part by '.'
    std::string last_part = parts.back();
    parts.pop_back();
    size_t dot_pos = last_part.find('.');
    if (dot_pos != std::string::npos) {
        parts.push_back(last_part.substr(0, dot_pos));
        parts.push_back(last_part.substr(dot_pos + 1)); // extension
    } else {
        parts.push_back(last_part);
    }

    // Check if we have enough parts
    if (parts.size() < 4) {
        fprintf(stderr, "Model path does not contain expected pattern\n");
        exit(EXIT_FAILURE);
    }

    // Get total files from [-2] position (before the extension)
    int total_files = std::stoi(parts[parts.size() - 2]);

    // Get base path by joining all parts except -start-of-end.gguf
    std::string base_path;
    for (size_t i = 0; i < parts.size() - 4; i++) {
        if (i > 0) {
            base_path += "-";
        }
        base_path += parts[i];
    }

    for (int i = 1; i <= total_files; i++) {
        char numbered_path[1024];
        snprintf(numbered_path, sizeof(numbered_path), "%s-%05d-of-%05d.gguf",
                base_path.c_str(), i, total_files);

        files.push_back({numbered_path, load_file_into_memory(numbered_path)});
    }

    return files;
}
}  // namespace

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);

    // Manually load into a memory buffer first
    std::vector<file_entry> files = load_files_into_memory(model_path);

    llama_backend_init();
    auto params              = llama_model_params{};
    params.use_mmap          = false;
    params.progress_callback = [](float progress, void * ctx) {
        (void) ctx;
        fprintf(stderr, "%.2f%% ", progress * 100.0f);
        // true means: Don't cancel the load
        return true;
    };

    printf("Loading model from %zu files\n", files.size());

    std::vector<const char*> file_paths;
    for (const auto& file : files) {
        printf("Found file %s with %zu bytes\n", file.path.c_str(), file.buffer.size());
        file_paths.push_back(file.path.c_str());
    }

    const char * async_load_context = "test-model-load";
    std::thread  fulfill_thread([&files, &async_load_context]() {
        for (const auto & file : files) {
            const bool success = llama_model_load_fulfill_split_future(file.path.c_str(), async_load_context,
                                                                       file.buffer.data(), file.buffer.size());
            printf("Fulfilling file %s: %s\n", file.path.c_str(), success ? "success" : "failure");
            if(!success) {
                exit(EXIT_FAILURE);
            }
        }
    });
    fprintf(stderr, "Loading model from splits\n");
    auto * model = llama_model_load_from_split_futures(
        file_paths.data(), file_paths.size(), async_load_context, params);
    fulfill_thread.join();

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
