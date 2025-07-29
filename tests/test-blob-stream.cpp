
#include <cstddef>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include <algorithm>

#include "llama-cpp.h"
#include "load_into_memory.h"
#include "blobs-uint8-buff-stream.h"

int main(int argc, char * argv[]) {
    (void)argc;  // Mark as unused
    (void)argv;  // Mark as unused

    // Create a vector of 1024 elements with known values
    std::vector<uint8_t> original_data(1024);
    for (size_t i = 0; i < original_data.size(); ++i) {
        original_data[i] = static_cast<uint8_t>(i % 256);
    }

    // Random number generator for blob sizes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(8, 24);

    // Create OwnedUint8BlobsStream with the blobs
    OwnedUint8BlobsStream blobs_stream(split_into_random_blobs(original_data, gen, size_dist));
    std::basic_istream<uint8_t> blobs_istream(&blobs_stream);

    // Create a direct streambuf for comparison
    std::vector<uint8_t> direct_data = original_data; // Copy for direct comparison
    Uint8BufferStreamBuf direct_buf(std::move(direct_data));
    std::basic_istream<uint8_t> direct_istream(&direct_buf);

    // Test random seeks and reads
    std::uniform_int_distribution<> pos_dist(0, original_data.size() - 1);
    std::uniform_int_distribution<> read_size_dist(1, 50);
    std::uniform_int_distribution<> consecutive_reads_dist(0, 10);

    for (int test = 0; test < 100; ++test) {
        // Random position to seek to
        size_t seek_pos = pos_dist(gen);

        // Random number of bytes to read
        size_t read_size = std::min(static_cast<size_t>(read_size_dist(gen)), original_data.size() - seek_pos);

        blobs_istream.clear();
        blobs_istream.seekg(seek_pos);

        direct_istream.clear();
        direct_istream.seekg(seek_pos);

        const int consecutive_reads_gen = consecutive_reads_dist(gen);
        std::size_t total_read = 0;
        for(int consecutive_read = 0; consecutive_read < consecutive_reads_gen; consecutive_read++) {
            std::vector<uint8_t> blobs_read(read_size);
            blobs_istream.read(blobs_read.data(), read_size);
            size_t blobs_bytes_read = blobs_istream.gcount();

            // Seek and read from full-buffer stream
            std::vector<uint8_t> direct_read(read_size);
            direct_istream.read(direct_read.data(), read_size);
            size_t direct_bytes_read = direct_istream.gcount();

            // Verify results match
            assert(blobs_bytes_read == direct_bytes_read);
            total_read += read_size;

            for (size_t i = 0; i < blobs_bytes_read; ++i) {
                assert(blobs_read[i] == direct_read[i]);
            }
        }
        if (test % 20 == 0) {
            std::cout << "Test " << test << ": seek to " << seek_pos
                << ", read " << total_read << " bytes (consecutive_reads=" << consecutive_reads_gen << ") - PASSED" << std::endl;
        }
    }

    // Test edge cases
    std::cout << "\nTesting edge cases..." << std::endl;

    // Test seeking to end
    blobs_istream.clear();
    blobs_istream.seekg(0, std::ios::end);
    std::streampos blobs_istream_size = blobs_istream.tellg();
    size_t original_data_size = original_data.size();
    assert(blobs_istream_size == static_cast<std::streampos>(original_data_size));

    // Test seeking from current position
    blobs_istream.clear();
    blobs_istream.seekg(100);
    blobs_istream.seekg(50, std::ios::cur);
    assert(blobs_istream.tellg() == static_cast<std::streampos>(150));

    // Test reading past end
    blobs_istream.clear();
    blobs_istream.seekg(original_data.size() - 10);
    std::vector<uint8_t> end_read(20);
    blobs_istream.read(end_read.data(), 20);
    assert(blobs_istream.gcount() == 10);

    std::cout << "All tests PASSED!" << std::endl;
    return EXIT_SUCCESS;
}
