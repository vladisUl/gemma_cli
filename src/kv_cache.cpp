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
