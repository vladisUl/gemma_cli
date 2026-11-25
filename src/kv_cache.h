#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>

namespace gemma4 {
enum class KvLayout { Key, Value };

// Update one token position in an INT8 KV cache without changing any other slot.
// K [channels, capacity]: strided, V [capacity, channels]: contiguous.
void update_kv_slot(std::vector<uint8_t>& cache,
                    const std::vector<uint8_t>& slice,
                    size_t channels, size_t capacity,
                    size_t position, KvLayout layout);

// Stage 6.5 Prefill-128 -> Decode cache block bridge. Only valid_rows
// are copied; all padding rows beyond the actual prompt are ignored.
// K input [C,source_rows] -> cache [C,capacity]: channel-strided.
// V input [source_rows,C] -> cache [capacity,C]: contiguous rows.
void update_kv_block(std::vector<uint8_t>& cache,
                     const std::vector<uint8_t>& block,
                     size_t channels,size_t capacity,size_t first_position,
                     size_t source_rows,size_t valid_rows,KvLayout layout);

// Identify the only modified cache position, or nullopt if identical.
// Reject differences at multiple positions (not a single-step update).
std::optional<size_t> find_changed_kv_slot(
    const std::vector<uint8_t>& before, const std::vector<uint8_t>& after,
    size_t channels, size_t capacity, KvLayout layout);
}
