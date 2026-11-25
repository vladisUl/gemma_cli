#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gemma4 {
struct CpuAuxTensor {
    std::string qnn_name;
    std::vector<uint8_t> bytes;
};

// CPU TFLite 2.14 C API; no TensorFlow development headers required.
// Executes decode_rope and decode_mask on the same Section1 model.
// Does not touch any QNN model, file or graph buffer.
std::vector<CpuAuxTensor> run_decode_aux(
    const std::string& model_file, int32_t position, int32_t token_id,
    const std::string& cpu_library =
        "/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1");
}
