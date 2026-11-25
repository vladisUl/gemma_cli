#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gemma4 {
// Compute original Gemma 4 Decode embedding from packed INT2 Section0
// or packed INT4 Section2 model. Uses read-only mmap, no TFLite runtime.
// Returns exactly 6144 (section 0) or 35840 (section 2) FLOAT32 bytes.
std::vector<uint8_t> compute_packed_embedding(
    const std::string& model_file, int section, int token);
} // namespace gemma4
