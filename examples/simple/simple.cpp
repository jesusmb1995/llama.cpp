#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <chrono>
#include <thread>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n Optional environment variables: LLAMA_EXAMPLE_MEMORY_BUFFER LLAMA_EXAMPLE_MEMORY_BUFFER_SPLIT");
    printf("\n");
}

namespace {
std::pair<std::uint8_t*, size_t> load_file_into_memory(const char * const model_path) {
    std::ifstream file_stream(model_path, std::ios::binary | std::ios::ate);
    if (!file_stream) {
        fprintf(stderr, "Failed to open file %s for reading into buffer\n", model_path);
        exit(EXIT_FAILURE);
    }

    const size_t file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    static_assert(sizeof(std::uint8_t) == sizeof(char), "uint8_t must be same size as char");
    std::uint8_t* buffer = new std::uint8_t[file_size];
    if (!file_stream.read((char*) buffer, file_size)) {
        fprintf(stderr, "Failed to read entire file into buffer\n");
        exit(EXIT_FAILURE);
    }

    return {buffer, file_size};
}

struct file_entry {
    std::string path;
    std::pair<std::uint8_t*, size_t> buffer;
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

file_entry load_tensor_list_file(const char * const model_path) {
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

    // Get base path by joining all parts except -start-of-end.gguf
    std::string base_path;
    for (size_t i = 0; i < parts.size() - 4; i++) {
        if (i > 0) {
            base_path += "-";
        }
        base_path += parts[i];
    }

    // Construct tensor list file path
    std::string tensor_list_path = base_path + ".tensors.txt";

    printf("Loading tensor list file: %s\n", tensor_list_path.c_str());
    return {tensor_list_path, load_file_into_memory(tensor_list_path.c_str())};
}
}  // namespace

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model;

    std::chrono::steady_clock::time_point load_start_time;
    if (getenv("LLAMA_EXAMPLE_MEMORY_BUFFER")) {
        auto buffer = load_file_into_memory(model_path.c_str());
        fprintf(stdout, "%s: loading model from memory buffer of size %zu\n", __func__, buffer.second);
        load_start_time = std::chrono::steady_clock::now();
        model           = llama_model_load_from_buffer(buffer.first, buffer.second, model_params);
    } else if (getenv("LLAMA_EXAMPLE_MEMORY_BUFFER_SPLIT")) {
        file_entry tensor_list_file = load_tensor_list_file(model_path.c_str());
        std::vector<file_entry> files = load_files_into_memory(model_path.c_str());
        fprintf(stdout, "%s: loading model from %zu file buffers\n", __func__, files.size());

        std::vector<const char *> file_paths;
        for (const auto & file : files) {
            printf("Found file %s with %zu bytes\n", file.path.c_str(), file.buffer.second);
            file_paths.push_back(file.path.c_str());
        }

        load_start_time                 = std::chrono::steady_clock::now();
        const char * async_load_context = "test-model-load";
        std::thread  fulfill_thread([&files, &tensor_list_file, &async_load_context]() {
            const bool success = llama_model_load_fulfill_split_future(tensor_list_file.path.c_str(), async_load_context,
                                                                        tensor_list_file.buffer.first, tensor_list_file.buffer.second);
            printf("Fulfilling tensor list file %s: %s\n", tensor_list_file.path.c_str(), success ? "success" : "failure");
            if (!success) {
                exit(EXIT_FAILURE);
            }

            for (const auto & file : files) {
                const bool success = llama_model_load_fulfill_split_future(file.path.c_str(), async_load_context,
                                                                            file.buffer.first, file.buffer.second);
                printf("Fulfilling file %s: %s\n", file.path.c_str(), success ? "success" : "failure");
                if (!success) {
                    exit(EXIT_FAILURE);
                }
            }
        });
        fprintf(stderr, "Loading model from splits\n");
        model =
            llama_model_load_from_split_futures(file_paths.data(), file_paths.size(), async_load_context,
                                                tensor_list_file.path.c_str(), model_params);
        fulfill_thread.join();
    } else {
        load_start_time = std::chrono::steady_clock::now();
        model           = llama_model_load_from_file(model_path.c_str(), model_params);
    }

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }
    std::chrono::steady_clock::time_point load_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> load_duration = load_end_time - load_start_time;
    fprintf(stdout, "%s: loading model took %f seconds\n", __func__, load_duration.count());

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
