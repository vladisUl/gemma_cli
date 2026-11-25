#include "kv_cache.h"
#include <algorithm>
#include <limits>
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
}
