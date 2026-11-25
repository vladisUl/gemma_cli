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

// Original TFLite 2.14 Prefill Section1: 128 RoPE positions and two
// query-padding-corrected causal masks (signed INT8). Read-only model.
// QNN runner strictly compares all six results to saved reference inputs.
std::vector<CpuAuxTensor> run_prefill_aux(
    const std::string& model_file,const std::vector<int32_t>& token_ids,
    const std::string& cpu_library =
        "/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1");

}
