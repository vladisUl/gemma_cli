#include "kv_cache.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace gemma4 {
void update_kv_slot(std::vector<uint8_t>& cache,
                    const std::vector<uint8_t>& slice,
                    size_t channels, size_t capacity,
                    size_t position, KvLayout layout) {
    if (!channels || !capacity || position >= capacity ||
        channels > std::numeric_limits<size_t>::max() / capacity ||
        cache.size() != channels * capacity || slice.size() != channels)
        throw std::invalid_argument("KV cache/slice dimensions or position invalid");
    switch (layout) {
        case KvLayout::Key:
            for (size_t c = 0; c < channels; ++c)
                cache[c * capacity + position] = slice[c];
            break;
        case KvLayout::Value:
            std::copy(slice.begin(), slice.end(), cache.begin() + position * channels);
            break;
        default:
            throw std::invalid_argument("Unknown KV layout");
    }
}

// Copy the first valid_rows of a [channels, source_rows] K block or
// [source_rows, channels] V block into a [channels, capacity] K cache
// or [capacity, channels] V cache, preserving every other slot.
// No padding rows beyond valid_rows are ever written.
void update_kv_block(std::vector<uint8_t>& cache,
                     const std::vector<uint8_t>& block,
                     size_t channels,size_t capacity,size_t first_position,
                     size_t source_rows,size_t valid_rows,KvLayout layout) {
    if(channels==0 || capacity==0 || source_rows==0 ||
       valid_rows==0 || valid_rows>source_rows ||
       first_position>capacity || valid_rows>capacity-first_position ||
       channels>std::numeric_limits<size_t>::max()/capacity ||
       channels>std::numeric_limits<size_t>::max()/source_rows ||
       cache.size()!=channels*capacity ||
       block.size()!=channels*source_rows)
        throw std::invalid_argument("Invalid K/V block shapes, position or valid row count");
    if(layout==KvLayout::Key) {
        for(size_t channel=0;channel<channels;++channel) {
            const auto from=block.begin()+channel*source_rows;
            const auto to=cache.begin()+channel*capacity+first_position;
            std::copy_n(from,valid_rows,to);
        }
    } else if(layout==KvLayout::Value) {
        const auto to=cache.begin()+first_position*channels;
        std::copy_n(block.begin(),valid_rows*channels,to);
    } else {
        throw std::invalid_argument("Unknown K/V block layout");
    }
}

std::optional<size_t> find_changed_kv_slot(
    const std::vector<uint8_t>& before, const std::vector<uint8_t>& after,
    size_t channels, size_t capacity, KvLayout layout) {
    if (!channels || !capacity ||
        channels > std::numeric_limits<size_t>::max() / capacity ||
        before.size() != channels * capacity ||
        after.size() != before.size())
        throw std::invalid_argument("Invalid KV cache dimensions for slot inference");
    if (layout != KvLayout::Key && layout != KvLayout::Value)
        throw std::invalid_argument("Unknown KV layout");
    std::optional<size_t> position;
    for (size_t i = 0; i < before.size(); ++i) {
        if (before[i] == after[i]) continue;
        const size_t p = layout == KvLayout::Key ? i % capacity : i / channels;
        if (position && *position != p)
            throw std::invalid_argument("KV cache changed at multiple token slots");
        position = p;
    }
    return position;
}
}
