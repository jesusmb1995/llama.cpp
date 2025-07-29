#pragma once

#include <streambuf>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <vector>

/// @brief Custom streambuf for uint8_t that handles multiple blobs sequentially
class Uint8BlobsStream : public std::basic_streambuf<uint8_t> {
public:
    Uint8BlobsStream(std::vector<std::pair<uint8_t*, std::size_t>> _blobs) : blobs(std::move(_blobs)) {
        if (!blobs.empty()) {
            current_blob = 0;
            current_blob_offset = 0;
            update_get_area();
            total_size = get_total_size(blobs);
        }
    }

protected:
    int_type underflow() override {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }

        // Try to move to next blob
        if (current_blob < blobs.size() - 1) {
            current_blob++;
            current_blob_offset = 0;
            update_get_area();
            if (gptr() < egptr()) {
                return traits_type::to_int_type(*gptr());
            }
        }

        return traits_type::eof();
    }

    std::streamsize xsgetn(char_type* s, std::streamsize n) override {
        std::streamsize total_read = 0;

        while (n > 0) {
            std::streamsize available = egptr() - gptr();
            std::streamsize to_read = std::min(n, available);

            if (to_read > 0) {
                std::memcpy(s + total_read, gptr(), to_read);
                setg(eback(), gptr() + to_read, egptr());
                current_blob_offset += to_read;
                total_read += to_read;
                n -= to_read;
                continue;
            }

            if (current_blob < blobs.size() - 1) {
                current_blob++;
                current_blob_offset = 0;
                update_get_area();
            } else {
                // No more blobs available
                break;
            }
        }

        return total_read;
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override {
        if (!(which & std::ios_base::in) || blobs.empty()) {
            return pos_type(off_type(-1));
        }

        off_type target_offset;
        if (dir == std::ios_base::beg) {
            target_offset = off;
        } else if (dir == std::ios_base::cur) {
            target_offset = get_current_global_offset() + off;
        } else if (dir == std::ios_base::end) {
            target_offset = total_size + off;
        } else {
            return pos_type(off_type(-1));
        }

        return seek_to_global_offset(target_offset);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in) override {
        if (!(which & std::ios_base::in)) {
            return pos_type(off_type(-1));
        }
        return seek_to_global_offset(pos);
    }

private:
    std::vector<std::pair<uint8_t*, std::size_t>> blobs;
    size_t current_blob = 0, current_blob_offset = 0, total_size = 0;

    void update_get_area() {
        if (current_blob < blobs.size()) {
            const auto& blob = blobs[current_blob];
            setg(blob.first, blob.first + current_blob_offset, blob.first + blob.second);
        }
    }

    off_type get_current_global_offset() const {
        off_type offset = 0;
        for (size_t i = 0; i < current_blob; ++i) {
            offset += blobs[i].second;
        }
        offset += current_blob_offset;
        return offset;
    }

    static off_type get_total_size(const std::vector<std::pair<uint8_t*, std::size_t>>& blobs) {
        off_type total = 0;
        for (const auto& blob : blobs) {
            total += blob.second;
        }
        return total;
    }

    pos_type seek_to_global_offset(off_type target_offset) {
        if (target_offset < 0 || static_cast<size_t>(target_offset) > total_size) {
            return pos_type(off_type(-1));
        }

        off_type current_offset_global = 0;
        for (size_t i = 0; i < blobs.size(); ++i) {
            off_type next_offset = current_offset_global + blobs[i].second;
            if (target_offset <= next_offset) {
                current_blob = i;
                current_blob_offset = target_offset - current_offset_global;
                update_get_area();
                return target_offset;
            }
            current_offset_global = next_offset;
        }

        return pos_type(off_type(-1));
    }
};

/// @brief Owned version of Uint8BlobsStream that takes ownership of the data
class OwnedUint8BlobsStream : public Uint8BlobsStream {
public:
  OwnedUint8BlobsStream(std::vector<std::vector<uint8_t>> && data)
      :
      Uint8BlobsStream(convert_to_blobs(data)),
      // Asuming data.data() on heap position remains unchanged after the move
      owned_data(std::move(data)) {}

private:
    std::vector<std::vector<uint8_t>> owned_data;

    static std::vector<std::pair<uint8_t*, std::size_t>> convert_to_blobs(const std::vector<std::vector<uint8_t>>& data) {
        std::vector<std::pair<uint8_t*, std::size_t>> blobs;
        blobs.reserve(data.size());

        for (const auto& vec : data) {
            if (!vec.empty()) {
                blobs.emplace_back(const_cast<uint8_t*>(vec.data()), vec.size());
            }
        }
        return blobs;
    }
};
