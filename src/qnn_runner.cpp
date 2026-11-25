// Stage 2: execute a precompiled QNN context directly (without Genie).
// Uses QNN System metadata as the source of truth for graph I/O.
// Intended for QAIRT 2.47 on Radxa Q8B; first target is Gemma 4 LM Head.
#include <QNN/QnnInterface.h>
#include <QNN/System/QnnSystemInterface.h>
#include "kv_cache.h"
#include "cpu_aux.h"
#include "packed_embedder.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <optional>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <set>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(Qnn_ErrorHandle_t status, const std::string& operation) {
    if (status != QNN_SUCCESS) {
        throw std::runtime_error(operation + ": QNN error " + std::to_string(status));
    }
}

struct SharedLibrary {
    void* handle = nullptr;

    explicit SharedLibrary(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            const char* message = dlerror();
            throw std::runtime_error("dlopen(" + path + "): " +
                                     (message ? message : "unknown error"));
        }
    }

    ~SharedLibrary() { if (handle) dlclose(handle); }
    SharedLibrary(const SharedLibrary&) = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

    template <typename Function>
    Function symbol(const char* name) const {
        dlerror();
        auto fn = reinterpret_cast<Function>(dlsym(handle, name));
        if (const char* error = dlerror()) {
            throw std::runtime_error(std::string("dlsym(") + name + "): " + error);
        }
        return fn;
    }
};

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    const auto end = file.tellg();
    if (end < 0) throw std::runtime_error("Cannot determine file size: " + path);
    if (static_cast<uint64_t>(end) > std::numeric_limits<size_t>::max())
        throw std::runtime_error("File too large: " + path);
    std::vector<uint8_t> data(static_cast<size_t>(end));
    file.seekg(0, std::ios::beg);
    if (!data.empty() &&
        !file.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("Cannot read file: " + path);
    }
    return data;
}

void write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot create output: " + path);
    if (!bytes.empty())
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Failed writing output: " + path);
}

const char* tensor_name(const Qnn_Tensor_t& t) {
    switch (t.version) {
        case QNN_TENSOR_VERSION_1: return t.v1.name;
        case QNN_TENSOR_VERSION_2: return t.v2.name;
        default: throw std::runtime_error("Unsupported QNN tensor metadata version");
    }
}

Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t& t) {
    switch (t.version) {
        case QNN_TENSOR_VERSION_1: return t.v1.dataType;
        case QNN_TENSOR_VERSION_2: return t.v2.dataType;
        default: throw std::runtime_error("Unsupported QNN tensor metadata version");
    }
}

Qnn_TensorDataFormat_t tensor_format(const Qnn_Tensor_t& t) {
    switch (t.version) {
        case QNN_TENSOR_VERSION_1: return t.v1.dataFormat;
        case QNN_TENSOR_VERSION_2: return t.v2.dataFormat;
        default: throw std::runtime_error("Unsupported QNN tensor metadata version");
    }
}

uint32_t tensor_rank(const Qnn_Tensor_t& t) {
    switch (t.version) {
        case QNN_TENSOR_VERSION_1: return t.v1.rank;
        case QNN_TENSOR_VERSION_2: return t.v2.rank;
        default: throw std::runtime_error("Unsupported QNN tensor metadata version");
    }
}

const uint32_t* tensor_dims(const Qnn_Tensor_t& t) {
    switch (t.version) {
        case QNN_TENSOR_VERSION_1: return t.v1.dimensions;
        case QNN_TENSOR_VERSION_2: return t.v2.dimensions;
        default: throw std::runtime_error("Unsupported QNN tensor metadata version");
    }
}

const Qnn_QuantizeParams_t& tensor_quant(const Qnn_Tensor_t& t) {
    switch (t.version) {
        case QNN_TENSOR_VERSION_1: return t.v1.quantizeParams;
        case QNN_TENSOR_VERSION_2: return t.v2.quantizeParams;
        default: throw std::runtime_error("Unsupported QNN tensor metadata version");
    }
}

unsigned dtype_bits(Qnn_DataType_t type) {
    switch (type) {
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8: return 8;
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16: return 16;
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32: return 32;
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_FLOAT_64: return 64;
        default: return 0; // Reject packed/unknown formats rather than guessing.
    }
}

std::string dtype_name(Qnn_DataType_t type) {
    switch (type) {
        case QNN_DATATYPE_INT_8: return "int8";
        case QNN_DATATYPE_UINT_8: return "uint8";
        case QNN_DATATYPE_SFIXED_POINT_8: return "sfix8";
        case QNN_DATATYPE_UFIXED_POINT_8: return "ufix8";
        case QNN_DATATYPE_INT_16: return "int16";
        case QNN_DATATYPE_UINT_16: return "uint16";
        case QNN_DATATYPE_FLOAT_16: return "fp16";
        case QNN_DATATYPE_INT_32: return "int32";
        case QNN_DATATYPE_UINT_32: return "uint32";
        case QNN_DATATYPE_FLOAT_32: return "fp32";
        case QNN_DATATYPE_INT_64: return "int64";
        case QNN_DATATYPE_UINT_64: return "uint64";
        case QNN_DATATYPE_FLOAT_64: return "fp64";
        default: return "qnn-type-" + std::to_string(static_cast<int>(type));
    }
}

size_t tensor_nbytes(const Qnn_Tensor_t& tensor) {
    if (tensor_format(tensor) != QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER)
        throw std::runtime_error("Only flat-buffer I/O is supported");
    const auto bits = dtype_bits(tensor_dtype(tensor));
    if (bits == 0 || (bits % 8) != 0)
        throw std::runtime_error("Unsupported tensor dtype: " + dtype_name(tensor_dtype(tensor)));
    const uint32_t rank = tensor_rank(tensor);
    const uint32_t* dims = tensor_dims(tensor);
    if (rank && !dims) throw std::runtime_error("Tensor dimensions are null");
    uint64_t count = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (dims[i] == 0)
            throw std::runtime_error("Dynamic/zero tensor dimension is unsupported");
        if (count > std::numeric_limits<uint64_t>::max() / dims[i])
            throw std::runtime_error("Tensor dimensions overflow");
        count *= dims[i];
    }
    const uint64_t bytes_per_element = bits / 8;
    if (count > std::numeric_limits<uint32_t>::max() / bytes_per_element)
        throw std::runtime_error("Tensor exceeds QNN raw client buffer size (uint32)");
    return static_cast<size_t>(count * bytes_per_element);
}

void bind_raw(Qnn_Tensor_t& tensor, std::vector<uint8_t>& data) {
    if (data.empty() || data.size() > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Invalid QNN tensor raw buffer size");
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        tensor.v1.clientBuf.data = data.data();
        tensor.v1.clientBuf.dataSize = static_cast<uint32_t>(data.size());
    } else if (tensor.version == QNN_TENSOR_VERSION_2) {
        tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
        tensor.v2.clientBuf.data = data.data();
        tensor.v2.clientBuf.dataSize = static_cast<uint32_t>(data.size());
        if (tensor.v2.isDynamicDimensions) {
            for (uint32_t i = 0; i < tensor.v2.rank; ++i) {
                if (tensor.v2.isDynamicDimensions[i])
                    throw std::runtime_error("Dynamic dimensions aren't supported");
            }
        }
    } else {
        throw std::runtime_error("Unsupported QNN tensor version");
    }
}

void describe_tensor(const Qnn_Tensor_t& t, const char* direction, uint32_t index) {
    std::cout << "  " << direction << "[" << index << "] "
              << (tensor_name(t) ? tensor_name(t) : "(unnamed)") << " "
              << dtype_name(tensor_dtype(t)) << " shape=[";
    const auto rank = tensor_rank(t);
    const auto dims = tensor_dims(t);
    for (uint32_t i = 0; i < rank; ++i)
        std::cout << (i ? "," : "") << (dims ? dims[i] : 0);
    std::cout << "]";
    try { std::cout << " bytes=" << tensor_nbytes(t); }
    catch (const std::exception& e) { std::cout << " bytes=unsupported(" << e.what() << ")"; }
    const auto& quant = tensor_quant(t);
    if (quant.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        std::cout << " scale=" << std::setprecision(10) << quant.scaleOffsetEncoding.scale
                  << " offset=" << quant.scaleOffsetEncoding.offset;
    }
    std::cout << "\n";
}

struct GraphView {
    std::string name;
    const Qnn_Tensor_t* inputs = nullptr;
    uint32_t num_inputs = 0;
    const Qnn_Tensor_t* outputs = nullptr;
    uint32_t num_outputs = 0;
};

GraphView get_graph(const QnnSystemContext_GraphInfo_t& g) {
    GraphView view;
    switch (g.version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
            view.name = g.graphInfoV1.graphName ? g.graphInfoV1.graphName : "";
            view.inputs = g.graphInfoV1.graphInputs;
            view.num_inputs = g.graphInfoV1.numGraphInputs;
            view.outputs = g.graphInfoV1.graphOutputs;
            view.num_outputs = g.graphInfoV1.numGraphOutputs;
            break;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
            view.name = g.graphInfoV2.graphName ? g.graphInfoV2.graphName : "";
            view.inputs = g.graphInfoV2.graphInputs;
            view.num_inputs = g.graphInfoV2.numGraphInputs;
            view.outputs = g.graphInfoV2.graphOutputs;
            view.num_outputs = g.graphInfoV2.numGraphOutputs;
            break;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
            view.name = g.graphInfoV3.graphName ? g.graphInfoV3.graphName : "";
            view.inputs = g.graphInfoV3.graphInputs;
            view.num_inputs = g.graphInfoV3.numGraphInputs;
            view.outputs = g.graphInfoV3.graphOutputs;
            view.num_outputs = g.graphInfoV3.numGraphOutputs;
            break;
        default:
            throw std::runtime_error("Unsupported QNN graph info version " +
                                     std::to_string(static_cast<int>(g.version)));
    }
    if ((view.num_inputs && !view.inputs) || (view.num_outputs && !view.outputs))
        throw std::runtime_error("Incomplete graph metadata: " + view.name);
    return view;
}

struct SystemMetadata {
    SharedLibrary library;
    const QnnSystemInterface_t* provider = nullptr;
    QnnSystemContext_Handle_t handle = nullptr;
    std::vector<GraphView> graphs;

    explicit SystemMetadata(const std::string& path) : library(path) {
        const auto getter = library.symbol<decltype(&QnnSystemInterface_getProviders)>(
            "QnnSystemInterface_getProviders");
        const QnnSystemInterface_t** providers = nullptr;
        uint32_t count = 0;
        check(getter(&providers, &count), "QnnSystemInterface_getProviders");
        if (!providers || !count) throw std::runtime_error("No QNN System providers");
        provider = providers[0];
    }

    ~SystemMetadata() {
        if (handle && provider && provider->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree)
            provider->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(handle);
    }

    void parse(std::vector<uint8_t>& binary) {
        const auto& api = provider->QNN_SYSTEM_INTERFACE_VER_NAME;
        if (!api.systemContextCreate || !api.systemContextGetBinaryInfo ||
            !api.systemContextFree)
            throw std::runtime_error("QNN System provider lacks context binary metadata APIs");
        check(api.systemContextCreate(&handle), "systemContextCreate");
        const QnnSystemContext_BinaryInfo_t* info = nullptr;
        Qnn_ContextBinarySize_t size = 0;
        check(api.systemContextGetBinaryInfo(handle, binary.data(), binary.size(),
                                             &info, &size),
              "systemContextGetBinaryInfo");
        if (!info) throw std::runtime_error("Missing QNN binary metadata");
        uint32_t num_graphs = 0;
        const QnnSystemContext_GraphInfo_t* graph_info = nullptr;
        switch (info->version) {
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
                num_graphs = info->contextBinaryInfoV1.numGraphs;
                graph_info = info->contextBinaryInfoV1.graphs;
                break;
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
                num_graphs = info->contextBinaryInfoV2.numGraphs;
                graph_info = info->contextBinaryInfoV2.graphs;
                break;
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
                num_graphs = info->contextBinaryInfoV3.numGraphs;
                graph_info = info->contextBinaryInfoV3.graphs;
                break;
            default:
                throw std::runtime_error("Unsupported QNN binary info version " +
                                         std::to_string(static_cast<int>(info->version)));
        }
        if (!num_graphs || !graph_info) throw std::runtime_error("No graphs in QNN context");
        for (uint32_t i = 0; i < num_graphs; ++i) graphs.push_back(get_graph(graph_info[i]));
    }

    const GraphView& find(const std::string& name) const {
        for (const auto& g : graphs)
            if (g.name == name) return g;
        throw std::runtime_error("Graph not found in binary metadata: " + name);
    }

    void inspect() const {
        std::cout << "QNN_META_OK: graphs=" << graphs.size() << "\n";
        for (const auto& g : graphs) {
            std::cout << "GRAPH " << g.name << " inputs=" << g.num_inputs
                      << " outputs=" << g.num_outputs << "\n";
            for (uint32_t i = 0; i < g.num_inputs; ++i)
                describe_tensor(g.inputs[i], "input", i);
            for (uint32_t i = 0; i < g.num_outputs; ++i)
                describe_tensor(g.outputs[i], "output", i);
        }
    }
};

struct QnnRuntime {
    SharedLibrary library;
    const QnnInterface_t* provider = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    std::vector<Qnn_ContextHandle_t> contexts;
    bool device_created = false;

    explicit QnnRuntime(const std::string& path) : library(path) {
        const auto getter =
            library.symbol<decltype(&QnnInterface_getProviders)>("QnnInterface_getProviders");
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        check(getter(&providers, &count), "QnnInterface_getProviders");
        if (!providers || !count) throw std::runtime_error("No QNN backend providers");
        provider = providers[0];
        const auto& api = provider->QNN_INTERFACE_VER_NAME;
        if (!api.backendCreate || !api.backendFree || !api.contextCreateFromBinary ||
            !api.contextFree || !api.graphRetrieve || !api.graphExecute)
            throw std::runtime_error("QNN backend lacks required execution APIs");
        check(api.backendCreate(nullptr, nullptr, &backend), "backendCreate");
        if (api.deviceCreate) {
            const auto code = api.deviceCreate(nullptr, nullptr, &device);
            if (code == QNN_SUCCESS) device_created = true;
            else if (code != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) check(code, "deviceCreate");
        }
    }

    ~QnnRuntime() {
        if (!provider) return;
        const auto& api = provider->QNN_INTERFACE_VER_NAME;
        if (api.contextFree) {
            for (auto iter = contexts.rbegin(); iter != contexts.rend(); ++iter)
                if (*iter) api.contextFree(*iter, nullptr);
        }
        if (device_created && device && api.deviceFree) api.deviceFree(device);
        if (backend && api.backendFree) api.backendFree(backend);
    }

    Qnn_GraphHandle_t load(const std::vector<uint8_t>& bytes, const std::string& name) {
        const auto& api = provider->QNN_INTERFACE_VER_NAME;
        if (bytes.empty()) throw std::runtime_error("Empty QNN context binary");
        Qnn_ContextHandle_t context = nullptr;
        check(api.contextCreateFromBinary(backend, device, nullptr, bytes.data(),
                                           static_cast<Qnn_ContextBinarySize_t>(bytes.size()),
                                           &context, nullptr),
              "contextCreateFromBinary(" + name + ")");
        // Keep every context live; its graph handle remains valid until destruction.
        contexts.push_back(context);
        Qnn_GraphHandle_t graph = nullptr;
        check(api.graphRetrieve(context, name.c_str(), &graph), "graphRetrieve(" + name + ")");
        return graph;
    }

    void execute(Qnn_GraphHandle_t graph,
                 const std::vector<Qnn_Tensor_t>& inputs,
                 std::vector<Qnn_Tensor_t>& outputs) {
        const auto& api = provider->QNN_INTERFACE_VER_NAME;
        check(api.graphExecute(graph, inputs.data(), static_cast<uint32_t>(inputs.size()),
                                outputs.data(), static_cast<uint32_t>(outputs.size()),
                                nullptr, nullptr),
              "QnnGraph_execute");
    }
};

struct Options {
    std::string binary, backend, system, graph, input, output, reference;
    std::string input_list, inputs_dir, outputs_dir, reference_dir, jobs, kv_next_list;
    std::string kv_next_reference_dir, kv_next_head_reference_dir;
    std::string cpu_aux_model;
    std::string cpu_embed_section0, cpu_embed_section2;
    std::string cpu_prefill_tokens, cpu_prefill_section0, cpu_prefill_section2;
    std::string cpu_prefill_aux_model;
    bool inspect = false;
    bool check_inputs = false;
    bool chain_decode_head = false;
    bool kv_feedback = false;
    bool kv_auto_pos = false;
    bool kv_run_next = false;
    bool cpu_aux_pos_set = false;
    bool cpu_aux_token_set = false;
    bool cpu_aux_next_pos_set = false;
    bool cpu_aux_next_token_set = false;
    bool cpu_embed_token_set = false;
    bool cpu_embed_next_token_set = false;
    bool prefill_kv_probe = false;
    bool prefill_kv_run = false;
    bool prefill_kv_token_set = false;
    bool prefill_kv_steps_set = false;
    bool prefill_seed_last_prompt = false;
    bool prefill_seed_first_prompt = false;
    bool prefill_bos_only_compare = false;
    bool prefill_first_input_compare = false;
    bool decode_only_prompt = false;
    bool compare_kv_histories = false;
    bool prefill_tie_policy_set = false;
    bool prefill_allow_tie_lowest = false;
    bool prefill_stop_on_eos = false;
    bool prefill_kv_diagnose = false;
    uint32_t kv_feedback_pos = 0;
    int32_t cpu_aux_pos = 0;
    int32_t cpu_aux_token = 0;
    int32_t cpu_aux_next_pos = 0;
    int32_t cpu_aux_next_token = 0;
    int32_t cpu_embed_token = 0;
    int32_t cpu_embed_next_token = 0;
    int32_t prefill_kv_token = 0;
    uint32_t prefill_kv_steps = 1;
    uint32_t repeat = 1;
};

namespace fs = std::filesystem;

// QNN and qnn-net-run use the original tensor name in "..._native.raw".
std::string native_basename(const Qnn_Tensor_t& tensor, uint32_t index) {
    const char* name = tensor_name(tensor);
    if (!name || !*name)
        return "tensor_" + std::to_string(index) + "_native.raw";
    std::string sanitized;
    for (const unsigned char c : std::string(name)) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
            sanitized += static_cast<char>(c);
        } else {
            sanitized += '_';
        }
    }
    if (sanitized.size() > 210) {
        // Keep the tensor index to prevent collisions when truncating names.
        sanitized = sanitized.substr(0, 190) + "_" + std::to_string(index);
    }
    return sanitized + "_native.raw";
}

std::vector<fs::path> parse_qnn_input_list(const std::string& file,
                                            const GraphView& graph) {
    std::ifstream stream(file);
    if (!stream) throw std::runtime_error("Cannot open --input-list: " + file);
    std::string line, entries;
    while (std::getline(stream, line)) {
        const auto begin = line.find_first_not_of(" \t\r");
        if (begin == std::string::npos || line[begin] == '#') continue;
        if (!entries.empty())
            throw std::runtime_error("--input-list has more than one sample/line; "
                                     "this runner executes exactly one sample");
        entries = line;
    }
    std::istringstream input(entries);
    std::vector<fs::path> paths;
    std::string token;
    while (input >> token) {
        if (paths.size() >= graph.num_inputs)
            throw std::runtime_error("--input-list has more paths than graph inputs");
        // qnn-net-run accepts both 'file.raw' and 'tensor_name:=file.raw'.
        const auto separator = token.find(":=");
        if (separator != std::string::npos) {
            const std::string expected =
                tensor_name(graph.inputs[paths.size()]) ?
                tensor_name(graph.inputs[paths.size()]) : "";
            if (token.substr(0, separator) != expected)
                throw std::runtime_error("Named input in --input-list is not in QNN "
                                         "metadata order: " + token.substr(0, separator) +
                                         " vs " + expected);
            token = token.substr(separator + 2);
        }
        if (token.empty()) throw std::runtime_error("Empty input-list path");
        // Match qnn-net-run convention: relative input paths are relative to CWD.
        paths.emplace_back(token);
    }
    if (paths.size() != graph.num_inputs)
        throw std::runtime_error("--input-list has " + std::to_string(paths.size()) +
                                 " files, graph requires " +
                                 std::to_string(graph.num_inputs));
    return paths;
}

std::vector<fs::path> resolve_inputs_dir(const std::string& dirname,
                                          const GraphView& graph) {
    const fs::path dir(dirname);
    if (!fs::is_directory(dir))
        throw std::runtime_error("Input directory does not exist: " + dirname);
    std::vector<fs::path> paths;
    for (uint32_t i = 0; i < graph.num_inputs; ++i) {
        const auto native = native_basename(graph.inputs[i], i);
        std::vector<fs::path> candidates{
            dir / native,
            dir / (native.substr(0, native.size() - 11) + ".raw"),
            dir / ("input_" + std::to_string(i) + ".raw")
        };
        std::vector<fs::path> found;
        for (const auto& candidate : candidates) {
            if (fs::is_regular_file(candidate) &&
                std::find(found.begin(), found.end(), candidate) == found.end())
                found.push_back(candidate);
        }
        if (found.size() != 1)
            throw std::runtime_error("Input[" + std::to_string(i) +
                                     "] " + (tensor_name(graph.inputs[i]) ?
                                            tensor_name(graph.inputs[i]) : "(unnamed)") +
                                     ": expected exactly one .raw file in " + dirname +
                                     ", found " + std::to_string(found.size()) +
                                     ". Use --input-list to specify exact paths.");
        paths.push_back(found[0]);
    }
    return paths;
}

// Defined before run_multi so references are resolved at compile time.
void compare_reference(const std::vector<uint8_t>& actual, const std::string& file);

void usage(const char* executable) {
    std::cout
        << "Direct QNN graph executor (QAIRT 2.47; without Genie)\n"
        << "Usage:\n"
        << "  " << executable << " --binary FILE --system LIB [--graph NAME] --inspect\n"
        << "  " << executable << " --binary FILE --system LIB --backend LIB --graph NAME \\\n"
        << "      --input FILE.raw --output FILE.raw [--reference FILE.raw] [--repeat N]\n"
        << "For multi-tensor graphs (e.g. Decode 38/31):\n"
        << "  " << executable << " --binary FILE --system LIB --backend LIB --graph NAME \\\n"
        << "      --input-list QNN_INPUT_LIST.txt --outputs-dir DIR [--reference-dir Result_0]\n"
        << "  or --inputs-dir DIR instead of --input-list FILE\n"
        << "  Add --check-inputs to validate all input tensors without NPU execution.\n"
        << "Persistent multi-context replay (one backend/device, all contexts kept loaded):\n"
        << "  " << executable << " --system LIB --backend LIB --jobs JOBS.txt\n"
        << "  JOBS.txt lines: BINARY GRAPH INPUT_LIST REFERENCE_RESULT_0 (no spaces in paths).\n"
        << "  Add --chain-decode-head to execute LM Head again, directly using Decode hidden-state RAM.\n"
        << "  Add --kv-feedback-pos N to update 30 K/V buffers in RAM (N=0..4095).\n"
        << "  Optional --kv-next-list FILE compares updated buffers with saved next-step caches.\n"
        << "  Or --kv-auto-pos with --kv-next-list to infer the changed KV slot.\n"
        << "  Add --kv-run-next to execute Decode again with updated KV and next-step non-KV inputs.\n"
        << "  Optional --kv-next-reference-dir Result_0 checks all 31 next Decode outputs.\n"
        << "  Optional --kv-next-head-reference-dir Result_0 checks next LM Head logits.\n"
        << "  --cpu-aux-model Section1.tflite --cpu-aux-pos N --cpu-aux-token ID\n"
        << "     replaces six first-Decode INT8 mask/RoPE inputs from CPU LiteRT;\n"
        << "     requires exact match to saved inputs before NPU execution.\n"
        << "  With --kv-run-next, --cpu-aux-next-pos N --cpu-aux-next-token ID\n"
        << "     replaces six next Decode AUX inputs via CPU in RAM.\n"
        << "  --cpu-embed-section0 FILE --cpu-embed-section2 FILE --cpu-embed-token ID\n"
        << "     replaces two FLOAT32 Decode embeddings with packed INT2/INT4 CPU results.\n"
        << "  Add --cpu-embed-next-token ID with --kv-run-next for the second Decode.\n"
        << "  --prefill-kv-probe builds 30 padded-aware Decode KV caches in scratch RAM\n"
        << "     from verified Prefill-128 outputs; validates mapping and quantization.\n"
        << "     Does NOT replace actual Decode KV inputs or execute a linked Decode.\n"
        << "  --prefill-kv-run --prefill-kv-token ID: separately run linked Decode+Head\n"
        << "     with Prefill KV cache (17 tokens) and CPU non-KV for position 17.\n"
        << "  Optional --prefill-kv-steps N (1..4; up to 16 with EOS-stop):\n"
        << "     previous Decode output KV feedback and prior LM Head INT8 argmax.\n"
        << "     Probe token is NOT derived from Prefill logits; no tokenizer.\n"
        << "  --prefill-stop-on-eos: LF107 continuation; EOS ids 1,106,50 stop.\n"
        << "  --prefill-tie-policy stop|lowest-id (default stop): print all tied\n"
        << "     INT8 max IDs; lowest-id allows an EXPERIMENTAL tie-break only.\n"
        << "  --prefill-seed-last-prompt instead of --prefill-kv-token: replays the\n"
        << "     final real prompt token at pos16 with Prefill KV rows 0..15,\n"
        << "     then computes the FIRST next token with linked LM Head.\n"
        << "     Experimental: outputs are NOT compared to independent saved Decode.\n"
        << "  --prefill-seed-first-prompt: one-step independent Decode(pos0)\n"
        << "     using the first verified prompt ID and zero initial KV; compare\n"
        << "     its KV slices to Prefill row0 (diagnostic only).\n"
        << "  --compare-kv-histories: Stage 6.15 same Decode(pos16/token4368)\n"
        << "     different KV prefixes, same CPU non-KV and same LM Head.\n"
        << "  --decode-only-prompt: Stage 6.14, consume 17 verified tokens\n"
        << "     using Decode only, feed 30 KV caches, run final LM Head.\n"
        << "  --prefill-first-input-compare: Stage 6.13 compare 8 first-token\n"
        << "     Prefill vs Decode non-KV input slices, including mask semantics.\n"
        << "  --prefill-bos-only-compare: Stage 6.12 A/B/C KV row0 control.\n"
        << "     Isolated Prefill BOS-only execution; needs first-token diagnostic.\n"
        << "  --prefill-kv-diagnose: read-only per-layer/row/layer-mapping INT8\n"
        << "     differences of last prompt Prefill KV vs Decode KV slices.\n"
        << "     Requires --prefill-seed-last-prompt; does NOT change any data.\n"
        << "  --cpu-prefill-tokens TOKENS_I32.raw --cpu-prefill-section0 MODEL0.tflite\n"
        << "  --cpu-prefill-section2 MODEL2.tflite --cpu-prefill-aux-model MODEL1.tflite\n"
        << "     computes all eight Prefill-128 non-KV inputs from CPU in RAM.\n"
        << "     Strictly compares to original saved Prefill inputs before QNN.\n"
        << "No implicit tensor conversions or synthetic zero inputs.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--inspect") {
            options.inspect = true;
            continue;
        }
        if (arg == "--check-inputs") {
            options.check_inputs = true;
            continue;
        }
        if (arg == "--chain-decode-head") {
            options.chain_decode_head = true;
            continue;
        }
        if (arg == "--prefill-kv-probe") {
            options.prefill_kv_probe = true;
            continue;
        }
        if (arg == "--prefill-stop-on-eos") {
            options.prefill_stop_on_eos = true;
            continue;
        }
        if (arg == "--prefill-kv-run") {
            options.prefill_kv_run = true;
            continue;
        }
        if (arg == "--prefill-seed-last-prompt") {
            options.prefill_seed_last_prompt = true;
            continue;
        }
        if (arg == "--prefill-seed-first-prompt") {
            options.prefill_seed_first_prompt = true;
            continue;
        }
        if (arg == "--compare-kv-histories") {
            options.compare_kv_histories = true;
            continue;
        }
        if (arg == "--decode-only-prompt") {
            options.decode_only_prompt = true;
            continue;
        }
        if (arg == "--prefill-first-input-compare") {
            options.prefill_first_input_compare = true;
            continue;
        }
        if (arg == "--prefill-bos-only-compare") {
            options.prefill_bos_only_compare = true;
            continue;
        }
        if (arg == "--prefill-kv-diagnose") {
            options.prefill_kv_diagnose = true;
            continue;
        }
        if (arg == "--kv-run-next") {
            options.kv_run_next = true;
            continue;
        }
        if (arg == "--kv-auto-pos") {
            if (options.kv_feedback)
                throw std::runtime_error("Choose --kv-auto-pos OR --kv-feedback-pos");
            options.kv_feedback = true;
            options.kv_auto_pos = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--binary") options.binary = value;
        else if (arg == "--backend") options.backend = value;
        else if (arg == "--system") options.system = value;
        else if (arg == "--graph") options.graph = value;
        else if (arg == "--input") options.input = value;
        else if (arg == "--output") options.output = value;
        else if (arg == "--reference") options.reference = value;
        else if (arg == "--input-list") options.input_list = value;
        else if (arg == "--inputs-dir") options.inputs_dir = value;
        else if (arg == "--outputs-dir") options.outputs_dir = value;
        else if (arg == "--reference-dir") options.reference_dir = value;
        else if (arg == "--jobs") options.jobs = value;
        else if (arg == "--kv-next-list") options.kv_next_list = value;
        else if (arg == "--kv-next-reference-dir") options.kv_next_reference_dir = value;
        else if (arg == "--kv-next-head-reference-dir") options.kv_next_head_reference_dir = value;
        else if (arg == "--cpu-aux-model") options.cpu_aux_model = value;
        else if (arg == "--cpu-prefill-tokens") options.cpu_prefill_tokens = value;
        else if (arg == "--cpu-prefill-section0") options.cpu_prefill_section0 = value;
        else if (arg == "--cpu-prefill-section2") options.cpu_prefill_section2 = value;
        else if (arg == "--cpu-prefill-aux-model") options.cpu_prefill_aux_model = value;
        else if (arg == "--prefill-kv-token") {
            if (options.prefill_kv_token_set || value.empty() || value.front()=='-')
                throw std::runtime_error("--prefill-kv-token requires ID 0..262143");
            size_t end=0;
            const auto id=std::stoul(value,&end);
            if(end!=value.size() || id>=262144)
                throw std::runtime_error("--prefill-kv-token out of range");
            options.prefill_kv_token=static_cast<int32_t>(id);
            options.prefill_kv_token_set=true;
        }
        else if (arg == "--prefill-tie-policy") {
            if(options.prefill_tie_policy_set ||
               (value!="stop" && value!="lowest-id"))
                throw std::runtime_error(
                    "--prefill-tie-policy must be stop or lowest-id exactly once");
            options.prefill_tie_policy_set=true;
            options.prefill_allow_tie_lowest=(value=="lowest-id");
        }
        else if (arg == "--prefill-kv-steps") {
            if(options.prefill_kv_steps_set || value.empty() || value.front()=='-')
                throw std::runtime_error("--prefill-kv-steps must be 1..16");
            size_t end=0;
            const auto steps=std::stoul(value,&end);
            if(end!=value.size() || steps<1 || steps>16)
                throw std::runtime_error("--prefill-kv-steps must be 1..16");
            options.prefill_kv_steps=static_cast<uint32_t>(steps);
            options.prefill_kv_steps_set=true;
        }
        else if (arg == "--cpu-aux-pos") {
            if (options.cpu_aux_pos_set || value.empty() || value.front() == '-')
                throw std::runtime_error("--cpu-aux-pos expects a single integer 0..4095");
            size_t consumed=0;
            const auto pos=std::stoul(value,&consumed);
            if (consumed!=value.size() || pos >= 4096)
                throw std::runtime_error("--cpu-aux-pos must be 0..4095");
            options.cpu_aux_pos=static_cast<int32_t>(pos);
            options.cpu_aux_pos_set=true;
        }
        else if (arg == "--cpu-aux-token") {
            if (options.cpu_aux_token_set || value.empty() || value.front() == '-')
                throw std::runtime_error("--cpu-aux-token expects a single integer 0..262143");
            size_t consumed=0;
            const auto token=std::stoul(value,&consumed);
            if (consumed!=value.size() || token >= 262144)
                throw std::runtime_error("--cpu-aux-token must be 0..262143");
            options.cpu_aux_token=static_cast<int32_t>(token);
            options.cpu_aux_token_set=true;
        }
        else if (arg == "--cpu-aux-next-pos") {
            if (options.cpu_aux_next_pos_set || value.empty() || value.front() == '-')
                throw std::runtime_error("--cpu-aux-next-pos expects an integer 0..4095");
            size_t consumed=0;
            const auto pos=std::stoul(value,&consumed);
            if (consumed!=value.size() || pos >= 4096)
                throw std::runtime_error("--cpu-aux-next-pos must be 0..4095");
            options.cpu_aux_next_pos=static_cast<int32_t>(pos);
            options.cpu_aux_next_pos_set=true;
        }
        else if (arg == "--cpu-aux-next-token") {
            if (options.cpu_aux_next_token_set || value.empty() || value.front() == '-')
                throw std::runtime_error("--cpu-aux-next-token expects an integer 0..262143");
            size_t consumed=0;
            const auto token=std::stoul(value,&consumed);
            if (consumed!=value.size() || token >= 262144)
                throw std::runtime_error("--cpu-aux-next-token must be 0..262143");
            options.cpu_aux_next_token=static_cast<int32_t>(token);
            options.cpu_aux_next_token_set=true;
        }
        else if (arg == "--cpu-embed-section0") options.cpu_embed_section0 = value;
        else if (arg == "--cpu-embed-section2") options.cpu_embed_section2 = value;
        else if (arg == "--cpu-embed-token" || arg == "--cpu-embed-next-token") {
            bool& set = arg == "--cpu-embed-token" ?
                options.cpu_embed_token_set : options.cpu_embed_next_token_set;
            int32_t& target = arg == "--cpu-embed-token" ?
                options.cpu_embed_token : options.cpu_embed_next_token;
            if (set || value.empty() || value.front() == '-')
                throw std::runtime_error(arg + " expects a token ID 0..262143");
            size_t consumed = 0;
            const auto number = std::stoul(value, &consumed);
            if (consumed != value.size() || number >= 262144)
                throw std::runtime_error(arg + " must be 0..262143");
            target = static_cast<int32_t>(number);
            set = true;
        }
        else if (arg == "--kv-feedback-pos") {
            if (options.kv_feedback)
                throw std::runtime_error("Choose --kv-auto-pos OR --kv-feedback-pos");
            if (value.empty() || value.front() == '-')
                throw std::runtime_error("--kv-feedback-pos must be 0..4095");
            size_t n = 0;
            const auto pos = std::stoul(value, &n);
            if (n != value.size() || pos >= 4096)
                throw std::runtime_error("--kv-feedback-pos must be 0..4095");
            options.kv_feedback = true;
            options.kv_feedback_pos = static_cast<uint32_t>(pos);
        }
        else if (arg == "--repeat") {
            size_t pos = 0;
            const auto repeat = std::stoul(value, &pos);
            if (pos != value.size() || repeat == 0 ||
                repeat > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("--repeat must be a positive integer");
            options.repeat = static_cast<uint32_t>(repeat);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (!options.jobs.empty()) {
        if (options.system.empty() || options.backend.empty())
            throw std::runtime_error("--jobs needs --system and --backend");
        if (options.cpu_aux_model.empty() != !options.cpu_aux_pos_set ||
            options.cpu_aux_model.empty() != !options.cpu_aux_token_set)
            throw std::runtime_error(
                "CPU AUX requires all of --cpu-aux-model, --cpu-aux-pos and --cpu-aux-token");
        if (options.cpu_aux_next_pos_set != options.cpu_aux_next_token_set)
            throw std::runtime_error(
                "CPU AUX next step requires both --cpu-aux-next-pos and --cpu-aux-next-token");
        if (options.cpu_aux_next_pos_set &&
            (options.cpu_aux_model.empty() || !options.cpu_aux_pos_set ||
             !options.cpu_aux_token_set || !options.kv_run_next))
            throw std::runtime_error(
                "CPU AUX next step requires --kv-run-next and first-step --cpu-aux-model/pos/token");
        const bool any_embed = !options.cpu_embed_section0.empty() ||
            !options.cpu_embed_section2.empty() || options.cpu_embed_token_set;
        const bool full_embed = !options.cpu_embed_section0.empty() &&
            !options.cpu_embed_section2.empty() && options.cpu_embed_token_set;
        if (any_embed && !full_embed)
            throw std::runtime_error(
                "CPU embeddings require --cpu-embed-section0, "
                "--cpu-embed-section2, and --cpu-embed-token");
        if (options.cpu_embed_next_token_set && (!full_embed || !options.kv_run_next))
            throw std::runtime_error(
                "--cpu-embed-next-token requires both embedding models, first token and --kv-run-next");
        const unsigned prefill_option_count =
            static_cast<unsigned>(!options.cpu_prefill_tokens.empty()) +
            static_cast<unsigned>(!options.cpu_prefill_section0.empty()) +
            static_cast<unsigned>(!options.cpu_prefill_section2.empty()) +
            static_cast<unsigned>(!options.cpu_prefill_aux_model.empty());
        if (prefill_option_count != 0 && prefill_option_count != 4)
            throw std::runtime_error(
                "CPU Prefill requires all four --cpu-prefill-* options");
        if (options.prefill_kv_probe && prefill_option_count != 4)
            throw std::runtime_error(
                "--prefill-kv-probe requires verified --cpu-prefill-tokens "
                "and all three CPU model files");
        if (options.prefill_kv_run &&
            (!options.prefill_kv_probe ||
             (static_cast<int>(options.prefill_kv_token_set) +
              static_cast<int>(options.prefill_seed_last_prompt) +
              static_cast<int>(options.prefill_seed_first_prompt) != 1) ||
             !options.chain_decode_head || options.cpu_aux_model.empty() ||
             options.cpu_embed_section0.empty() ||
             options.cpu_embed_section2.empty()))
            throw std::runtime_error(
                "--prefill-kv-run requires --prefill-kv-probe, exactly one "
                "of --prefill-kv-token ID, --prefill-seed-last-prompt, "
                "or --prefill-seed-first-prompt, "
                "--chain-decode-head, --cpu-aux-model, "
                "--cpu-embed-section0 and --cpu-embed-section2");
        if (options.prefill_kv_token_set && !options.prefill_kv_run)
            throw std::runtime_error(
                "--prefill-kv-token only valid with --prefill-kv-run");
        if (options.prefill_kv_steps_set && !options.prefill_kv_run)
            throw std::runtime_error(
                "--prefill-kv-steps only valid with --prefill-kv-run");
        if (options.prefill_seed_last_prompt && !options.prefill_kv_run)
            throw std::runtime_error(
                "--prefill-seed-last-prompt only valid with --prefill-kv-run");
        if (options.prefill_seed_first_prompt &&
            (!options.prefill_kv_run || options.prefill_kv_steps!=1))
            throw std::runtime_error(
                "--prefill-seed-first-prompt requires --prefill-kv-run "
                "and exactly one Decode step");
        if (options.prefill_tie_policy_set && !options.prefill_kv_run)
            throw std::runtime_error(
                "--prefill-tie-policy only valid with --prefill-kv-run");
        if (options.prefill_kv_diagnose &&
            (!options.prefill_kv_run ||
             !(options.prefill_seed_last_prompt ||
               options.prefill_seed_first_prompt)))
            throw std::runtime_error(
                "--prefill-kv-diagnose requires --prefill-kv-run "
                "and --prefill-seed-last-prompt or --prefill-seed-first-prompt");
        if (options.prefill_bos_only_compare &&
            (!options.prefill_seed_first_prompt || !options.prefill_kv_diagnose ||
             !options.prefill_kv_run || options.prefill_kv_steps!=1))
            throw std::runtime_error(
                "--prefill-bos-only-compare requires --prefill-seed-first-prompt, "
                "--prefill-kv-diagnose and one Decode step");
        if (options.prefill_first_input_compare &&
            !options.prefill_bos_only_compare)
            throw std::runtime_error(
                "--prefill-first-input-compare requires --prefill-bos-only-compare");
        if (options.decode_only_prompt &&
            (!options.prefill_first_input_compare ||
             !options.prefill_seed_first_prompt ||
             options.prefill_kv_steps!=1))
            throw std::runtime_error(
                "--decode-only-prompt requires Stage 6.13 first-input diagnostic "
                "and one Decode step");
        if (options.compare_kv_histories && !options.decode_only_prompt)
            throw std::runtime_error(
                "--compare-kv-histories requires --decode-only-prompt");
        if(options.prefill_stop_on_eos &&
           (!options.prefill_kv_run || !options.prefill_kv_token_set ||
            options.prefill_kv_token!=107 ||
            options.prefill_kv_steps<2 || options.prefill_allow_tie_lowest ||
            options.prefill_seed_last_prompt || options.prefill_seed_first_prompt))
            throw std::runtime_error(
                "--prefill-stop-on-eos requires --prefill-kv-run, "
                "--prefill-kv-token 107, 2..16 steps, non-arbitrary tie policy");
        if(options.prefill_kv_steps>4 && !options.prefill_stop_on_eos)
            throw std::runtime_error(
                "More than 4 --prefill-kv-steps requires --prefill-stop-on-eos");
        if (!options.kv_next_list.empty() && !options.kv_feedback)
            throw std::runtime_error("--kv-next-list requires --kv-feedback-pos or --kv-auto-pos");
        if (options.kv_auto_pos && options.kv_next_list.empty())
            throw std::runtime_error("--kv-auto-pos requires --kv-next-list");
        if (options.kv_run_next && (!options.kv_feedback || options.kv_next_list.empty()))
            throw std::runtime_error("--kv-run-next requires --kv-auto-pos or --kv-feedback-pos, and --kv-next-list");
        if (!options.kv_next_reference_dir.empty() && !options.kv_run_next)
            throw std::runtime_error("--kv-next-reference-dir requires --kv-run-next");
        if (!options.kv_next_head_reference_dir.empty() &&
            (!options.kv_run_next || !options.chain_decode_head))
            throw std::runtime_error("--kv-next-head-reference-dir requires --kv-run-next and --chain-decode-head");
        if (!options.binary.empty() || !options.graph.empty() ||
            !options.input.empty() || !options.output.empty() ||
            !options.reference.empty() || !options.input_list.empty() ||
            !options.inputs_dir.empty() || !options.outputs_dir.empty() ||
            !options.reference_dir.empty() || options.inspect ||
            options.check_inputs || options.repeat != 1)
            throw std::runtime_error("--jobs cannot be combined with single-graph options");
        return options;
    }
    if (options.prefill_kv_probe || options.prefill_kv_run ||
        options.prefill_kv_token_set || options.prefill_kv_steps_set ||
        options.prefill_seed_last_prompt || options.prefill_seed_first_prompt ||
        options.prefill_bos_only_compare || options.prefill_first_input_compare ||
        options.decode_only_prompt || options.compare_kv_histories ||
        options.prefill_tie_policy_set || options.prefill_stop_on_eos ||
        options.prefill_kv_diagnose || options.chain_decode_head ||
        options.kv_feedback || options.kv_run_next ||
        !options.kv_next_list.empty() || !options.kv_next_reference_dir.empty() ||
        !options.kv_next_head_reference_dir.empty() ||
        !options.cpu_aux_model.empty() || options.cpu_aux_pos_set ||
        options.cpu_aux_token_set || options.cpu_aux_next_pos_set ||
        options.cpu_aux_next_token_set ||
        !options.cpu_embed_section0.empty() ||
        !options.cpu_embed_section2.empty() ||
        options.cpu_embed_token_set || options.cpu_embed_next_token_set ||
        !options.cpu_prefill_tokens.empty() ||
        !options.cpu_prefill_section0.empty() ||
        !options.cpu_prefill_section2.empty() ||
        !options.cpu_prefill_aux_model.empty())
        throw std::runtime_error("Chained/KV/CPU AUX options require --jobs");
    if (options.binary.empty() || options.system.empty())
        throw std::runtime_error("--binary and --system are required");
    if (!options.inspect) {
        if (options.graph.empty())
            throw std::runtime_error("Execution needs --graph");
        const bool multi = !options.input_list.empty() || !options.inputs_dir.empty() ||
                           !options.outputs_dir.empty() || !options.reference_dir.empty();
        if (options.backend.empty() && !options.check_inputs)
            throw std::runtime_error("Execution needs --backend");
        if (options.check_inputs && !multi)
            throw std::runtime_error("--check-inputs requires multi-tensor mode");
        if (multi) {
            if (options.outputs_dir.empty())
                throw std::runtime_error("Multi-tensor mode needs --outputs-dir");
            if (options.input_list.empty() == options.inputs_dir.empty())
                throw std::runtime_error("Specify exactly one --input-list or --inputs-dir");
            if (!options.input.empty() || !options.output.empty() ||
                !options.reference.empty())
                throw std::runtime_error("Do not mix --input/--output with multi-tensor mode");
        } else {
            if (options.input.empty() || options.output.empty())
                throw std::runtime_error("Execution needs --input and --output");
            if (options.input == options.output ||
                (!options.reference.empty() && options.output == options.reference))
                throw std::runtime_error("Output path must differ from input/reference path");
        }
    }
    return options;
}

void compare_reference(const std::vector<uint8_t>& actual, const std::string& file) {
    if (file.empty()) return;
    const auto expected = read_file(file);
    if (expected.size() != actual.size())
        throw std::runtime_error("REFERENCE_MISMATCH: output has " +
                                 std::to_string(actual.size()) +
                                 " bytes; reference has " + std::to_string(expected.size()));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (expected[i] != actual[i])
            throw std::runtime_error("REFERENCE_MISMATCH at byte " + std::to_string(i) +
                                     ": actual=" + std::to_string(actual[i]) +
                                     ", expected=" + std::to_string(expected[i]));
    }
    std::cout << "REFERENCE_MATCH: " << actual.size() << " bytes identical\n";
}

void run_multi(const Options& options, const GraphView& graph,
               const std::vector<uint8_t>& binary) {
    if (!graph.num_inputs || !graph.num_outputs)
        throw std::runtime_error("Graph has no inputs/outputs");
    std::vector<fs::path> filenames =
        !options.input_list.empty() ?
            parse_qnn_input_list(options.input_list, graph) :
            resolve_inputs_dir(options.inputs_dir, graph);

    std::vector<std::vector<uint8_t>> input_buffers(graph.num_inputs);
    std::vector<Qnn_Tensor_t> inputs(graph.inputs, graph.inputs + graph.num_inputs);
    for (uint32_t i = 0; i < graph.num_inputs; ++i) {
        const size_t bytes = tensor_nbytes(inputs[i]);
        input_buffers[i] = read_file(filenames[i].string());
        if (input_buffers[i].size() != bytes)
            throw std::runtime_error("INPUT_SIZE_MISMATCH [" + std::to_string(i) +
                                     "] " + (tensor_name(inputs[i]) ?
                                            tensor_name(inputs[i]) : "(unnamed)") +
                                     ": expected " + std::to_string(bytes) +
                                     " bytes, got " +
                                     std::to_string(input_buffers[i].size()) +
                                     " from " + filenames[i].string());
        bind_raw(inputs[i], input_buffers[i]);
        std::cout << "INPUT_OK [" << i << "] " << filenames[i]
                  << " bytes=" << bytes << "\n";
    }

    if (options.check_inputs) {
        std::cout << "INPUTS_VALIDATED: " << graph.num_inputs
                  << " tensors; no NPU execution\n";
        return;
    }

    const fs::path outdir(options.outputs_dir);
    fs::create_directories(outdir);
    if (!options.inputs_dir.empty() &&
        fs::is_directory(options.inputs_dir) &&
        fs::equivalent(outdir, fs::path(options.inputs_dir)))
        throw std::runtime_error("--outputs-dir must differ from --inputs-dir");
    if (!options.reference_dir.empty() &&
        fs::is_directory(options.reference_dir) &&
        fs::equivalent(outdir, fs::path(options.reference_dir)))
        throw std::runtime_error("--outputs-dir must differ from --reference-dir");

    std::vector<std::vector<uint8_t>> output_buffers(graph.num_outputs);
    std::vector<Qnn_Tensor_t> outputs(graph.outputs, graph.outputs + graph.num_outputs);
    std::vector<fs::path> output_files;
    std::set<std::string> output_names;
    for (uint32_t i = 0; i < graph.num_outputs; ++i) {
        const size_t bytes = tensor_nbytes(outputs[i]);
        output_buffers[i].resize(bytes);
        bind_raw(outputs[i], output_buffers[i]);
        const auto name = native_basename(outputs[i], i);
        if (!output_names.insert(name).second)
            throw std::runtime_error("Duplicate output filename: " + name);
        output_files.push_back(outdir / name);
        std::cout << "OUTPUT_READY [" << i << "] " << output_files.back()
                  << " bytes=" << bytes << "\n";
    }

    QnnRuntime runtime(options.backend);
    const auto load_start = std::chrono::steady_clock::now();
    const Qnn_GraphHandle_t graph_handle = runtime.load(binary, graph.name);
    const auto exec_start = std::chrono::steady_clock::now();
    for (uint32_t iter = 0; iter < options.repeat; ++iter)
        runtime.execute(graph_handle, inputs, outputs);
    const auto exec_end = std::chrono::steady_clock::now();

    for (uint32_t i = 0; i < graph.num_outputs; ++i)
        write_file(output_files[i].string(), output_buffers[i]);

    const double load_ms =
        std::chrono::duration<double, std::milli>(exec_start - load_start).count();
    const double exec_ms =
        std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
    std::cout << std::fixed << std::setprecision(3)
              << "QNN_MULTI_EXECUTE_OK: " << graph.name
              << " inputs=" << graph.num_inputs
              << " outputs=" << graph.num_outputs
              << " load_ms=" << load_ms
              << " avg_execute_ms=" << exec_ms / options.repeat
              << " repeat=" << options.repeat << "\n";

    if (!options.reference_dir.empty()) {
        const fs::path refdir(options.reference_dir);
        if (!fs::is_directory(refdir))
            throw std::runtime_error("Reference directory does not exist: " +
                                     options.reference_dir);
        for (uint32_t i = 0; i < graph.num_outputs; ++i) {
            const auto expected = refdir / output_files[i].filename();
            std::cout << "REFERENCE [" << i << "] " << expected << "\n";
            compare_reference(output_buffers[i], expected.string());
        }
        std::cout << "ALL_REFERENCES_MATCH: " << graph.num_outputs
                  << " output tensors\n";
    }
}



// Stage 4 / smoke test: keep multiple independent QNN contexts live in one
// process. Saved input samples are replayed independently; this is NOT yet a
// tokenizer / KV-cache generation loop.
struct ReplayJob {
    std::string binary_path, graph_name, input_list, reference_dir;
    std::vector<uint8_t> binary;
    std::unique_ptr<SystemMetadata> metadata;
    Qnn_GraphHandle_t handle = nullptr;
    std::vector<std::vector<uint8_t>> input_buffers, output_buffers;
    std::vector<Qnn_Tensor_t> inputs, outputs;
    std::vector<fs::path> references;
};

std::vector<std::unique_ptr<ReplayJob>> parse_replay_jobs(const std::string& filename) {
    std::ifstream stream(filename);
    if (!stream) throw std::runtime_error("Cannot open --jobs: " + filename);
    std::vector<std::unique_ptr<ReplayJob>> jobs;
    std::string line;
    size_t number = 0;
    while (std::getline(stream, line)) {
        ++number;
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        auto job = std::make_unique<ReplayJob>();
        std::istringstream input(line);
        std::string extra;
        if (!(input >> job->binary_path >> job->graph_name >>
                  job->input_list >> job->reference_dir) || (input >> extra))
            throw std::runtime_error("Invalid jobs line " + std::to_string(number) +
                ": expected BINARY GRAPH INPUT_LIST REFERENCE_RESULT_0");
        if (jobs.size() >= 3)
            throw std::runtime_error("--jobs accepts at most 3 contexts in this smoke test");
        jobs.push_back(std::move(job));
    }
    if (jobs.empty()) throw std::runtime_error("--jobs has no graph entries");
    return jobs;
}

void prepare_replay_job(ReplayJob& job, const std::string& system_library) {
    job.binary = read_file(job.binary_path);
    job.metadata = std::make_unique<SystemMetadata>(system_library);
    job.metadata->parse(job.binary);
    const GraphView& graph = job.metadata->find(job.graph_name);
    if (!graph.num_inputs || !graph.num_outputs)
        throw std::runtime_error("Graph has no I/O: " + graph.name);
    const auto paths = parse_qnn_input_list(job.input_list, graph);

    job.inputs.assign(graph.inputs, graph.inputs + graph.num_inputs);
    job.outputs.assign(graph.outputs, graph.outputs + graph.num_outputs);
    job.input_buffers.resize(graph.num_inputs);
    job.output_buffers.resize(graph.num_outputs);
    const fs::path refdir(job.reference_dir);
    if (!fs::is_directory(refdir))
        throw std::runtime_error("Reference directory missing: " + job.reference_dir);
    for (uint32_t i = 0; i < graph.num_inputs; ++i) {
        const size_t required = tensor_nbytes(job.inputs[i]);
        job.input_buffers[i] = read_file(paths[i].string());
        if (job.input_buffers[i].size() != required)
            throw std::runtime_error("Wrong input size for " + job.graph_name + " [" +
                std::to_string(i) + "]: expected " + std::to_string(required) +
                ", actual " + std::to_string(job.input_buffers[i].size()) +
                " in " + paths[i].string());
        bind_raw(job.inputs[i], job.input_buffers[i]);
    }
    for (uint32_t i = 0; i < graph.num_outputs; ++i) {
        const size_t required = tensor_nbytes(job.outputs[i]);
        job.output_buffers[i].resize(required);
        bind_raw(job.outputs[i], job.output_buffers[i]);
        const fs::path reference = refdir / native_basename(job.outputs[i], i);
        if (!fs::is_regular_file(reference))
            throw std::runtime_error("Missing reference for " + job.graph_name + ": " +
                                     reference.string());
        if (fs::file_size(reference) != required)
            throw std::runtime_error("Wrong reference size for " + job.graph_name + ": " +
                                     reference.string());
        job.references.push_back(reference);
    }
    std::cout << "REPLAY_PREPARED: " << job.graph_name
              << " inputs=" << graph.num_inputs << " outputs=" << graph.num_outputs
              << " binary_bytes=" << job.binary.size() << "\n" << std::flush;
}

// Match Decode's final INT8 hidden state to LM Head input using *actual*
// QNN metadata; index/name guesses are not valid for all AOT exports.
size_t find_decode_head_link(const ReplayJob& decode, const ReplayJob& head) {
    if (decode.graph_name != "qnn_partition_0" ||
        head.graph_name != "qnn_partition_1")
        throw std::runtime_error("Chaining requires qnn_partition_0 and qnn_partition_1");
    if (head.inputs.size() != 1 || head.outputs.size() != 1)
        throw std::runtime_error("Chaining requires a one-input/one-output LM Head");
    const auto& target = head.inputs[0];
    if (tensor_dtype(target) != QNN_DATATYPE_SFIXED_POINT_8 ||
        tensor_nbytes(target) != 1536)
        throw std::runtime_error("Unexpected LM Head hidden-state signature");
    const auto& target_quant = tensor_quant(target);
    if (target_quant.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
        throw std::runtime_error("LM Head requires per-tensor scale/offset metadata");

    size_t result = decode.outputs.size();
    for (size_t i = 0; i < decode.outputs.size(); ++i) {
        const auto& candidate = decode.outputs[i];
        if (tensor_dtype(candidate) != tensor_dtype(target) ||
            tensor_rank(candidate) != tensor_rank(target) ||
            tensor_nbytes(candidate) != tensor_nbytes(target))
            continue;
        bool dims_match = true;
        for (uint32_t d = 0; d < tensor_rank(target); ++d) {
            if (tensor_dims(candidate)[d] != tensor_dims(target)[d]) {
                dims_match = false;
                break;
            }
        }
        if (!dims_match) continue;
        const auto& quant = tensor_quant(candidate);
        if (quant.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
            continue;
        if (quant.scaleOffsetEncoding.scale != target_quant.scaleOffsetEncoding.scale ||
            quant.scaleOffsetEncoding.offset != target_quant.scaleOffsetEncoding.offset)
            continue;
        if (result != decode.outputs.size())
            throw std::runtime_error("Ambiguous Decode hidden-state output: multiple metadata matches");
        result = i;
    }
    if (result == decode.outputs.size())
        throw std::runtime_error("No Decode output matches LM Head input shape/dtype/quantization");
    return result;
}

void execute_chained_decode_head(QnnRuntime& runtime,
                                 ReplayJob& decode,
                                 ReplayJob& head,
                                 size_t decode_output_index,
                                 bool linked_prefill = false) {
    auto& hidden = decode.output_buffers.at(decode_output_index);
    if (hidden.size() != tensor_nbytes(head.inputs[0]))
        throw std::runtime_error("Chained hidden-state size changed");
    // Pass the existing Decode output buffer directly as the LM Head input.
    // QNN graphExecute is synchronous; do not free or overwrite 'hidden' yet.
    std::vector<Qnn_Tensor_t> linked_inputs = head.inputs;
    bind_raw(linked_inputs[0], hidden);
    const auto start = std::chrono::steady_clock::now();
    runtime.execute(head.handle, linked_inputs, head.outputs);
    const auto stop = std::chrono::steady_clock::now();

    const auto& logits = head.output_buffers.at(0);
    if (tensor_dtype(head.outputs[0]) != QNN_DATATYPE_SFIXED_POINT_8 ||
        logits.size() != 262144)
        throw std::runtime_error("Unexpected LM Head output: expected 262144 signed INT8 logits");

    // Diagnostic ONLY: integer argmax is not a production sampler.
    // Several token IDs can share the highest value after INT8 quantization.
    int max_value = -129;
    size_t first_max = 0, ties = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const int q = logits[i] < 128 ? static_cast<int>(logits[i])
                                      : static_cast<int>(logits[i]) - 256;
        if (q > max_value) {
            max_value = q;
            first_max = i;
            ties = 1;
        } else if (q == max_value) {
            ++ties;
        }
    }

    std::cout << "CHAIN_DECODE_HEAD_OK: output_name="
              << (tensor_name(decode.outputs[decode_output_index]) ?
                  tensor_name(decode.outputs[decode_output_index]) : "(unnamed)")
              << " hidden_bytes=" << hidden.size()
              << " logits_bytes=" << logits.size()
              << " head_execute_ms="
              << std::chrono::duration<double, std::milli>(stop - start).count()
              << " int8_argmax_id=" << first_max
              << " max_logit_q=" << max_value
              << " argmax_ties=" << ties << "\n" << std::flush;
    if (linked_prefill) {
        std::ostringstream tied_ids;
        size_t seen=0;
        for(size_t i=0;i<logits.size();++i) {
            const int q=logits[i]<128?static_cast<int>(logits[i]):
                                          static_cast<int>(logits[i])-256;
            if(q==max_value) {
                tied_ids<<(seen?",":"")<<i;
                ++seen;
            }
        }
        if(seen!=ties)
            throw std::runtime_error("CHAIN_LINKED_INT8_MAX_TIE_COUNT_INVALID");
        std::cout<<"CHAIN_LINKED_INT8_MAX_TOKENS: max_logit_q="<<max_value
                 <<" ties="<<ties
                 <<" token_ids="<<tied_ids.str()
                 <<" quantized_logits_only=1\n"<<std::flush;
        std::cout<<"CHAIN_NOTE: logits are from an experimental linked "
                    "Prefill-cache Decode, NOT from any saved Decode or "
                    "independent LM Head reference.\n";
    } else {
        std::cout << "CHAIN_NOTE: logits are from the saved Decode sample, NOT the "
                     "independent LM Head reference sample; do not compare them as "
                     "if both samples were identical.\n";
    }
}

// These mapping indices come from original Gemma 4 Decode TFLite signature,
// NOT the QNN array order. QNN retains "StatefulPartitionedCall_<index>"
// output names, while the actual graph output array is permuted. This known
// AOT layout is checked strictly by name, shape, dtype, and input quantization.
struct KvLink {
    size_t input_index = 0;
    size_t output_index = 0;
    bool key = false;
    uint32_t channels = 0;
    std::string name;
};

std::vector<KvLink> decode_kv_links(const ReplayJob& decode) {
    if (decode.graph_name != "qnn_partition_0")
        throw std::runtime_error("KV feedback requires qnn_partition_0");
    constexpr std::array<uint32_t, 15> order{
        0,1,10,11,12,13,14,2,3,4,5,6,7,8,9
    };
    std::vector<KvLink> links;
    std::set<size_t> seen_input, seen_output;
    for (uint32_t n = 0; n < 30; ++n) {
        const bool key = n < 15;
        const std::string name = "decode_kv_cache_" +
            std::string(key ? "k_" : "v_") +
            std::to_string(order[n % 15]) + "_0";
        const std::string output = "StatefulPartitionedCall_" + std::to_string(n);
        size_t input_index = decode.inputs.size();
        size_t output_index = decode.outputs.size();
        for (size_t i = 0; i < decode.inputs.size(); ++i)
            if (tensor_name(decode.inputs[i]) &&
                name == tensor_name(decode.inputs[i]))
                input_index = i;
        for (size_t i = 0; i < decode.outputs.size(); ++i)
            if (tensor_name(decode.outputs[i]) &&
                output == tensor_name(decode.outputs[i]))
                output_index = i;
        if (input_index == decode.inputs.size() ||
            output_index == decode.outputs.size())
            throw std::runtime_error("KV map missing " + name + " or " + output);
        const auto& ci = decode.inputs[input_index];
        const auto& so = decode.outputs[output_index];
        if (tensor_rank(ci) != 4 || tensor_rank(so) != 4 ||
            tensor_dtype(ci) != QNN_DATATYPE_SFIXED_POINT_8 ||
            tensor_dtype(so) != QNN_DATATYPE_SFIXED_POINT_8)
            throw std::runtime_error("KV dtype/rank mismatch: " + name);
        const auto* c = tensor_dims(ci);
        const auto* s = tensor_dims(so);
        const uint32_t channels = key ? c[2] : c[3];
        if (c[0] != 1 || c[1] != 1 ||
            (key ? c[3] : c[2]) != 4096 ||
            (channels != 256 && channels != 512) ||
            s[0] != 1 || s[1] != 1 ||
            (key ? s[2] : s[3]) != channels ||
            (key ? s[3] : s[2]) != 1 ||
            tensor_nbytes(ci) != static_cast<size_t>(channels) * 4096 ||
            tensor_nbytes(so) != channels)
            throw std::runtime_error("KV tensor shape mismatch: " + name);
        if (!seen_input.insert(input_index).second ||
            !seen_output.insert(output_index).second)
            throw std::runtime_error("Duplicate KV mapping: " + name);
        const auto& qcache = tensor_quant(ci);
        const auto& qslice = tensor_quant(so);
        if (qcache.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
            qslice.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
            (qcache.scaleOffsetEncoding.scale != qslice.scaleOffsetEncoding.scale ||
             qcache.scaleOffsetEncoding.offset != qslice.scaleOffsetEncoding.offset))
            throw std::runtime_error("KV quantization mismatch: " + name);
        links.push_back({input_index, output_index, key, channels, name});
    }
    if (seen_input.size() != 30 || seen_output.size() != 30)
        throw std::runtime_error("Expected exactly 30 unique KV mappings");
    return links;
}

uint32_t infer_decode_kv_position(
    const ReplayJob& decode,
    const std::vector<KvLink>& links,
    const std::vector<fs::path>& next_files) {
    if (next_files.size() != decode.inputs.size())
        throw std::runtime_error("Missing next Decode paths for KV auto-position");
    std::optional<size_t> inferred;
    size_t changed_caches = 0;
    for (const auto& link : links) {
        const auto& before = decode.input_buffers.at(link.input_index);
        const auto after = read_file(next_files.at(link.input_index).string());
        std::optional<size_t> slot;
        try {
            slot = gemma4::find_changed_kv_slot(
                before, after, link.channels, 4096,
                link.key ? gemma4::KvLayout::Key : gemma4::KvLayout::Value);
        } catch (const std::invalid_argument& ex) {
            throw std::runtime_error("KV_AUTO_POS_MISMATCH: " + link.name +
                                     ": " + ex.what());
        }
        if (!slot) continue;
        ++changed_caches;
        if (inferred && *inferred != *slot)
            throw std::runtime_error("KV_AUTO_POS_MISMATCH: changes in different "
                "slots across cache tensors; samples may not be consecutive: " +
                link.name);
        inferred = slot;
    }
    if (!inferred)
        throw std::runtime_error("KV_AUTO_POS_AMBIGUOUS: no changed cache bytes");
    std::cout << "KV_AUTO_POS_OK: pos=" << *inferred
              << " changed_cache_tensors=" << changed_caches
              << " total_cache_tensors=" << links.size()
              << "\n" << std::flush;
    return static_cast<uint32_t>(*inferred);
}

void apply_kv_feedback(ReplayJob& decode, const std::vector<KvLink>& links,
                       uint32_t pos, const std::vector<fs::path>& next_files) {
    size_t inserted = 0;
    for (const auto& link : links) {
        auto& cache = decode.input_buffers.at(link.input_index);
        const auto& slice = decode.output_buffers.at(link.output_index);
        gemma4::update_kv_slot(cache, slice, link.channels, 4096, pos,
                               link.key ? gemma4::KvLayout::Key : gemma4::KvLayout::Value);
        inserted += slice.size();
    }
    std::cout << "KV_FEEDBACK_OK: pos=" << pos
              << " cache_tensors=" << links.size()
              << " inserted_bytes=" << inserted
              << " layout=K_strided,V_contiguous\n" << std::flush;
    if (next_files.empty()) {
        std::cout << "KV_FEEDBACK_NOTE: next Decode cache was not provided; "
                     "this checks only in-RAM placement, not next-step equivalence\n";
        return;
    }
    for (const auto& link : links) {
        const auto& actual = decode.input_buffers.at(link.input_index);
        const fs::path& path = next_files.at(link.input_index);
        const auto expected = read_file(path.string());
        if (expected.size() != actual.size())
            throw std::runtime_error("KV_NEXT_SIZE_MISMATCH: " + link.name);
        auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        if (mismatch.first != actual.end())
            throw std::runtime_error("KV_NEXT_MISMATCH: " + link.name +
                " byte=" + std::to_string(
                    static_cast<size_t>(mismatch.first - actual.begin())) +
                " slot=" + std::to_string(pos) +
                "; verify samples are consecutive and slot is correct");
    }
    std::cout << "KV_NEXT_CACHE_MATCH: 30 cache tensors byte-identical\n" << std::flush;
}

// At this point all K/V inputs have been updated from the previous Decode
// outputs and (when available) byte-verified against next-step saved inputs.
// Replace ONLY the eight non-cache inputs (RoPE/masks/embeddings/per-layer)
// with the next saved sample. Never replace the computed KV-cache buffers
// with saved next-step files: that would bypass the feedback we want to test.
// CPU Section1 helper is defined below the two-step replay function.
void apply_cpu_aux_decode(ReplayJob& decode,
                          const std::string& model_file,
                          int32_t position, int32_t token_id,
                          const std::string& scope);

void apply_cpu_embeddings_decode(ReplayJob& decode, const Options& options,
                                 int32_t token_id, const std::string& scope);

void execute_next_decode(QnnRuntime& runtime,
                         ReplayJob& decode,
                         const std::vector<KvLink>& kv_links,
                         const std::vector<fs::path>& next_files,
                         const std::vector<fs::path>& next_refs,
                         const Options& options) {
    if (next_files.size() != decode.inputs.size() ||
        kv_links.size() != 30)
        throw std::runtime_error("Incomplete next-step Decode input metadata");
    std::set<size_t> kv_indices;
    for (const auto& link : kv_links) kv_indices.insert(link.input_index);
    if (kv_indices.size() != 30 || decode.inputs.size() != 38 ||
        decode.outputs.size() != 31)
        throw std::runtime_error("Unexpected Decode I/O count in two-step replay");

    size_t changed_non_kv = 0;
    for (size_t i = 0; i < decode.inputs.size(); ++i) {
        if (kv_indices.count(i)) continue;
        auto next_input = read_file(next_files.at(i).string());
        if (next_input.size() != tensor_nbytes(decode.inputs[i]))
            throw std::runtime_error("Next Decode input size mismatch: " +
                                     next_files.at(i).string());
        decode.input_buffers[i] = std::move(next_input);
        ++changed_non_kv;
    }
    if (options.cpu_aux_next_pos_set) {
        // The 30 K/V caches are already updated in RAM; only six out of the
        // eight non-KV inputs are substituted by Section1 CPU execution.
        // Compare against this exact saved next-step sample before NPU.
        apply_cpu_aux_decode(decode, options.cpu_aux_model,
                             options.cpu_aux_next_pos,
                             options.cpu_aux_next_token, "next");
    }
    if (options.cpu_embed_next_token_set) {
        // Preserve the original saved next-step embeddings solely as a
        // byte-exact reference. QNN receives freshly computed RAM buffers.
        apply_cpu_embeddings_decode(decode, options,
                                    options.cpu_embed_next_token, "next");
    }
    // Rebind ALL 38 input pointers, including the 30 updated K/V buffers,
    // in case replacing any non-KV vector caused its allocation to move.
    for (size_t i = 0; i < decode.inputs.size(); ++i)
        bind_raw(decode.inputs[i], decode.input_buffers[i]);

    if (changed_non_kv != 8)
        throw std::runtime_error("Expected 8 next-step non-KV inputs");
    std::cout << "DECODE_NEXT_INPUTS_READY: kv_from_RAM=" << kv_links.size()
              << " non_kv_from_saved_next=" << changed_non_kv
              << " cpu_aux_replaced=" << (options.cpu_aux_next_pos_set ? 6 : 0)
              << " cpu_embeddings_replaced="
              << (options.cpu_embed_next_token_set ? 2 : 0)
              << "\n" << std::flush;
    const auto start = std::chrono::steady_clock::now();
    runtime.execute(decode.handle, decode.inputs, decode.outputs);
    const auto stop = std::chrono::steady_clock::now();
    std::cout << "DECODE_NEXT_EXECUTE_OK: graph=" << decode.graph_name
              << " outputs=" << decode.outputs.size()
              << " execute_ms="
              << std::chrono::duration<double, std::milli>(stop - start).count()
              << "\n" << std::flush;
    if (next_refs.empty()) {
        std::cout << "DECODE_NEXT_NOTE: no next-step graph reference was supplied; "
                     "execution succeeded but outputs have not been compared\n";
        return;
    }
    if (next_refs.size() != decode.outputs.size())
        throw std::runtime_error("Missing next Decode graph references");
    // Compare the full output tensors, not just KV slices. This includes
    // the quantized 1536-byte hidden state destined for LM Head.
    for (size_t i = 0; i < next_refs.size(); ++i)
        compare_reference(decode.output_buffers[i], next_refs[i].string());
    std::cout << "DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical"
              << "\n" << std::flush;
}

// Before loading any QNN context: replace the six saved AUX inputs with
// tensors computed by local Linux/ARM64 TFLite Section1. Validate exact bytes
// against the original QNN Decode input buffers BEFORE replacement, so the
// pre-existing 31-output graph reference test remains independently meaningful.
void apply_cpu_aux_decode(ReplayJob& decode,
                          const std::string& model_file,
                          int32_t position, int32_t token_id,
                          const std::string& scope) {
    if (decode.graph_name != "qnn_partition_0" ||
        decode.inputs.size() != 38 || decode.input_buffers.size() != 38)
        throw std::runtime_error("CPU_AUX_REQUIRES_DECODE_38_INPUTS");
    auto outputs = gemma4::run_decode_aux(
        model_file, position, token_id);
    if (outputs.size() != 6)
        throw std::runtime_error("CPU_AUX_INVALID_OUTPUT_COUNT");
    std::set<size_t> indexes;
    size_t byte_total=0;
    for (auto& item : outputs) {
        size_t index=decode.inputs.size();
        for (size_t j=0;j<decode.inputs.size();++j) {
            const char* name=tensor_name(decode.inputs[j]);
            if (name && item.qnn_name == name) {
                if (index != decode.inputs.size())
                    throw std::runtime_error("CPU_AUX_DUPLICATE_QNN_TENSOR_NAME");
                index=j;
            }
        }
        if (index==decode.inputs.size() || !indexes.insert(index).second)
            throw std::runtime_error("CPU_AUX_QNN_TENSOR_NOT_UNIQUE: "+item.qnn_name);
        if (tensor_dtype(decode.inputs[index]) != QNN_DATATYPE_SFIXED_POINT_8 ||
            tensor_nbytes(decode.inputs[index]) != item.bytes.size())
            throw std::runtime_error("CPU_AUX_QNN_INPUT_TYPE_SIZE_MISMATCH: "+item.qnn_name);
        const auto& saved=decode.input_buffers[index];
        const auto mismatch=std::mismatch(item.bytes.begin(),item.bytes.end(),
                                          saved.begin());
        if (mismatch.first != item.bytes.end())
            throw std::runtime_error(
                "CPU_AUX_REFERENCE_MISMATCH: " + item.qnn_name +
                " first_byte=" +
                std::to_string(static_cast<size_t>(mismatch.first-item.bytes.begin())) +
                " (check token ID and position)");
        std::cout << "CPU_AUX_INPUT_MATCH: " << item.qnn_name
                  << " qnn_index=" << index
                  << " bytes=" << item.bytes.size() << "\n" << std::flush;
        byte_total += item.bytes.size();
        // This exact memory allocation becomes the actual QNN input.
        decode.input_buffers[index]=std::move(item.bytes);
        bind_raw(decode.inputs[index],decode.input_buffers[index]);
    }
    if(indexes.size()!=6 || byte_total!=9730)
        throw std::runtime_error("CPU_AUX_EXPECTED_6_INPUTS_9730_BYTES");
    std::cout << "CPU_AUX_QNN_REPLACED: tensors=" << indexes.size()
              << " bytes=" << byte_total
              << " from=Section1_TFLite_2.14"
              << " verified_against_saved_decode=1 scope=" << scope
              << "\n" << std::flush;
    if (scope == "next")
        std::cout << "CPU_AUX_NEXT_QNN_REPLACED: tensors=6 bytes=9730"
                  << " cpu_pos=" << position
                  << " cpu_token=" << token_id
                  << "\n" << std::flush;
}

void apply_cpu_aux_first_decode(ReplayJob& decode, const Options& options) {
    apply_cpu_aux_decode(decode, options.cpu_aux_model,
                         options.cpu_aux_pos, options.cpu_aux_token, "first");
}

// Stage 5.9: recompute both FLOAT32 embedding inputs directly in RAM.
// The source weights are original read-only packed Section0 INT2 / Section2
// INT4 TFLite files, not converted/approximated weights. Fail closed on
// any mismatch with the original saved Decode input bytes.
void apply_cpu_embeddings_decode(ReplayJob& decode, const Options& options,
                                 int32_t token_id, const std::string& scope) {
    if (decode.graph_name != "qnn_partition_0" || decode.inputs.size() != 38 ||
        decode.input_buffers.size() != 38)
        throw std::runtime_error("CPU_EMBED_REQUIRES_DECODE_38_INPUTS");
    struct Spec {
        const char* name;
        int section;
        size_t bytes;
        const std::string* model;
    };
    const std::array<Spec,2> specs{{
        {"decode_embeddings_0", 0, 6144, &options.cpu_embed_section0},
        {"decode_per_layer_embeddings_0", 2, 35840, &options.cpu_embed_section2},
    }};
    std::set<size_t> used;
    size_t bytes_total=0;
    for (const auto& spec:specs) {
        size_t index=decode.inputs.size();
        for (size_t i=0;i<decode.inputs.size();++i) {
            const char* name=tensor_name(decode.inputs[i]);
            if(name && spec.name==std::string(name)) {
                if(index!=decode.inputs.size())
                    throw std::runtime_error("CPU_EMBED_AMBIGUOUS_QNN_INPUT");
                index=i;
            }
        }
        if(index==decode.inputs.size() || !used.insert(index).second)
            throw std::runtime_error(std::string("CPU_EMBED_INPUT_MISSING: ")+spec.name);
        if(tensor_dtype(decode.inputs[index])!=QNN_DATATYPE_FLOAT_32 ||
           tensor_nbytes(decode.inputs[index])!=spec.bytes ||
           decode.input_buffers[index].size()!=spec.bytes)
            throw std::runtime_error(std::string("CPU_EMBED_INPUT_METADATA_MISMATCH: ")+spec.name);
        auto computed=gemma4::compute_packed_embedding(
            *spec.model, spec.section, token_id);
        if(computed.size()!=spec.bytes)
            throw std::runtime_error(std::string("CPU_EMBED_OUTPUT_SIZE_MISMATCH: ")+spec.name);
        const auto& reference=decode.input_buffers[index];
        const auto mismatch=std::mismatch(
            computed.begin(),computed.end(),reference.begin());
        if(mismatch.first!=computed.end())
            throw std::runtime_error(
                std::string("CPU_EMBED_REFERENCE_MISMATCH: ")+spec.name+
                " scope="+scope+" token="+std::to_string(token_id)+
                " first_byte="+std::to_string(
                    static_cast<size_t>(mismatch.first-computed.begin())));
        bytes_total+=computed.size();
        decode.input_buffers[index]=std::move(computed);
        bind_raw(decode.inputs[index],decode.input_buffers[index]);
        std::cout<<"CPU_EMBED_INPUT_MATCH: "<<spec.name
                 <<" qnn_index="<<index
                 <<" bytes="<<spec.bytes
                 <<" token="<<token_id
                 <<" scope="<<scope<<"\n"<<std::flush;
    }
    if(used.size()!=2 || bytes_total!=41984)
        throw std::runtime_error("CPU_EMBED_EXPECTED_TWO_INPUTS_41984_BYTES");
    std::cout<<"CPU_EMBED_QNN_REPLACED: scope="<<scope
             <<" token="<<token_id<<" tensors=2 bytes="<<bytes_total
             <<" verified_against_saved_decode=1\n"<<std::flush;
}

// Strict QNN metadata index and source reference validation for one CPU
// generated Prefill-128 tensor. The original saved .raw is NEVER modified;
// only the QNN in-memory vector is replaced after a byte-exact match.
void install_cpu_prefill_tensor(
    ReplayJob& prefill, size_t index, const std::string& required_name,
    Qnn_DataType_t required_type,size_t required_bytes,
    std::vector<uint8_t> generated) {
    if(prefill.graph_name!="qnn_partition_2" ||
       prefill.inputs.size()!=36 || prefill.input_buffers.size()!=36 ||
       index>=prefill.inputs.size())
        throw std::runtime_error("CPU_PREFILL_QNN_GRAPH_METADATA_MISMATCH");
    const char* name=tensor_name(prefill.inputs[index]);
    if(!name || required_name!=name ||
       tensor_dtype(prefill.inputs[index])!=required_type ||
       tensor_nbytes(prefill.inputs[index])!=required_bytes ||
       prefill.input_buffers[index].size()!=required_bytes ||
       generated.size()!=required_bytes)
        throw std::runtime_error("CPU_PREFILL_INPUT_METADATA_MISMATCH: "+
                                 required_name+" index="+std::to_string(index));
    const auto& saved=prefill.input_buffers[index];
    const auto diff=std::mismatch(generated.begin(),generated.end(),saved.begin());
    if(diff.first!=generated.end())
        throw std::runtime_error("CPU_PREFILL_INPUT_REFERENCE_MISMATCH: "+
            required_name+" index="+std::to_string(index)+" first_byte="+
            std::to_string(static_cast<size_t>(diff.first-generated.begin())));
    prefill.input_buffers[index]=std::move(generated);
    bind_raw(prefill.inputs[index],prefill.input_buffers[index]);
    std::cout<<"CPU_PREFILL_INPUT_MATCH: "<<required_name
             <<" qnn_index="<<index<<" bytes="<<required_bytes
             <<" from_CPU=1\n"<<std::flush;
}

void apply_cpu_prefill(ReplayJob& prefill,const Options& options) {
    if(prefill.graph_name!="qnn_partition_2" || prefill.inputs.size()!=36 ||
       prefill.outputs.size()!=30)
        throw std::runtime_error("CPU_PREFILL_REQUIRES_36_INPUT_30_OUTPUT_GRAPH");
    const auto token_raw=read_file(options.cpu_prefill_tokens);
    if(token_raw.size()!=128*sizeof(int32_t))
        throw std::runtime_error("CPU_PREFILL_TOKENS_REQUIRE_512_BYTES");
    std::vector<int32_t> tokens(128);
    std::memcpy(tokens.data(),token_raw.data(),token_raw.size());
    for(int32_t id:tokens)
        if(id<0 || id>=262144)
            throw std::runtime_error("CPU_PREFILL_TOKEN_OUT_OF_RANGE");
    std::cout<<"CPU_PREFILL_INPUTS_BEGIN: positions=128 "
             <<"source_tokens=verified_i32_512 from=Section0_Section1_Section2\n"
             <<std::flush;

    auto embed0=gemma4::compute_packed_prefill_embeddings(
        options.cpu_prefill_section0,0,tokens);
    install_cpu_prefill_tensor(prefill,32,"prefill_128_embeddings_0",
                              QNN_DATATYPE_FLOAT_32,786432,std::move(embed0));
    auto embed2=gemma4::compute_packed_prefill_embeddings(
        options.cpu_prefill_section2,2,tokens);
    install_cpu_prefill_tensor(prefill,7,"prefill_128_per_layer_embeddings_0",
                              QNN_DATATYPE_FLOAT_32,4587520,std::move(embed2));

    auto aux=gemma4::run_prefill_aux(options.cpu_prefill_aux_model,tokens);
    if(aux.size()!=6)
        throw std::runtime_error("CPU_PREFILL_AUX_EXPECTED_6_TENSORS");
    struct AuxTarget {
        const char* name;
        size_t index;
        size_t bytes;
    };
    const std::array<AuxTarget,6> spec{{
        {"prefill_128_pos_emb_cos_0",3,65536},
        {"prefill_128_pos_emb_sin_0",5,65536},
        {"prefill_128_pos_emb_local_sin_0",6,32768},
        {"prefill_128_pos_emb_local_cos_0",18,32768},
        {"prefill_128_mask_local_0",21,540672},
        {"prefill_128_mask_global_0",26,540672}
    }};
    size_t aux_bytes=0;
    for(size_t i=0;i<spec.size();++i) {
        if(aux[i].qnn_name!=spec[i].name)
            throw std::runtime_error("CPU_PREFILL_AUX_OUTPUT_ORDER_MISMATCH");
        aux_bytes+=aux[i].bytes.size();
        install_cpu_prefill_tensor(prefill,spec[i].index,spec[i].name,
              QNN_DATATYPE_SFIXED_POINT_8,spec[i].bytes,
              std::move(aux[i].bytes));
    }
    if(aux_bytes!=1277952)
        throw std::runtime_error("CPU_PREFILL_AUX_EXPECTED_1277952_BYTES");
    // Rebind every QNN input after vector moves; 28 saved initial K/V
    // buffers remain intact, while all 8 non-KV inputs come from CPU.
    for(size_t i=0;i<prefill.inputs.size();++i)
        bind_raw(prefill.inputs[i],prefill.input_buffers[i]);
    std::cout<<"CPU_PREFILL_ALL_NONKV_QNN_REPLACED: tensors=8"
             <<" embeddings_bytes=5373952 aux_bytes="<<aux_bytes
             <<" total_bytes="<<(5373952+aux_bytes)
             <<" kv_inputs_from_saved=28 verified_against_saved_qnn=1\n"
             <<std::flush;
}

// Stage 6.5 mapping probe: map all 30 Prefill KV blocks by their output
// names to the validated 30 Decode cache inputs, then copy ONLY the
// non-padding prompt rows into fresh zero-initialized scratch caches.
// This is a read-only diagnostic: it NEVER modifies active Decode buffers,
// source Prefill tensors or the independently verified original QNN samples.
void probe_prefill_kv_bridge(
    const ReplayJob& prefill,const ReplayJob& decode,
    const std::string& token_file,
    std::vector<std::vector<uint8_t>>* linked_caches = nullptr,
    bool seed_last_prompt = false,
    bool seed_first_prompt = false) {
    constexpr size_t kCapacity=4096,kBlockRows=128;
    if(prefill.graph_name!="qnn_partition_2" ||
       decode.graph_name!="qnn_partition_0" ||
       prefill.inputs.size()!=36 || prefill.outputs.size()!=30 ||
       decode.inputs.size()!=38 || decode.outputs.size()!=31)
        throw std::runtime_error("KV_PREFILL_PROBE_GRAPH_METADATA_INVALID");
    const auto token_data=read_file(token_file);
    if(token_data.size()!=kBlockRows*sizeof(int32_t))
        throw std::runtime_error("KV_PREFILL_PROBE_TOKEN_FILE_SIZE_INVALID");
    std::array<int32_t,kBlockRows> token_ids{};
    std::memcpy(token_ids.data(),token_data.data(),token_data.size());
    size_t valid_tokens=kBlockRows;
    for(size_t i=0;i<kBlockRows;++i) {
        const auto id=token_ids[i];
        if(id<0 || id>=262144)
            throw std::runtime_error("KV_PREFILL_PROBE_TOKEN_ID_INVALID");
        if(id==0 && valid_tokens==kBlockRows)valid_tokens=i;
        if(id!=0 && valid_tokens!=kBlockRows)
            throw std::runtime_error("KV_PREFILL_PROBE_PADDING_NOT_CONTIGUOUS");
    }
    if(valid_tokens==0)
        throw std::runtime_error("KV_PREFILL_PROBE_EMPTY_PROMPT");
    if(seed_last_prompt && (!linked_caches || valid_tokens<2))
        throw std::runtime_error(
            "PREFILL_LAST_TOKEN_PROBE_NEEDS_TWO_TOKENS_AND_SCRATCH");
    if(seed_first_prompt && (!linked_caches || !valid_tokens || seed_last_prompt))
        throw std::runtime_error(
            "PREFILL_FIRST_TOKEN_PROBE_REQUIRES_EXCLUSIVE_SCRATCH_MODE");
    // Last-token mode copies the true prompt prefix. First-token mode
    // deliberately copies NONE of the Prefill KV outputs so it can test
    // the numerical disagreement in the absence of history.
    const size_t copied_rows=seed_first_prompt?0:
        valid_tokens-(seed_last_prompt?1:0);
    const auto links=decode_kv_links(decode);
    if(links.size()!=30)
        throw std::runtime_error("KV_PREFILL_PROBE_REQUIRES_30_KV_LINKS");
    if(linked_caches)linked_caches->resize(decode.inputs.size());

    size_t prefill_input_kv=0,prefill_input_nonzero=0;
    uint64_t prefill_input_bytes=0;
    for(size_t i=0;i<prefill.inputs.size();++i) {
        const char* name=tensor_name(prefill.inputs[i]);
        if(!name || std::string(name).find("prefill_128_kv_cache_")!=0)
            continue;
        ++prefill_input_kv;
        if(linked_caches) {
            const auto& tensor=prefill.inputs[i];
            const auto& quant=tensor_quant(tensor);
            if(tensor_dtype(tensor)!=QNN_DATATYPE_SFIXED_POINT_8 ||
               quant.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
               quant.scaleOffsetEncoding.offset!=0)
                throw std::runtime_error(
                    "PREFILL_LINK_INITIAL_CACHE_ZERO_QUANT_UNSUPPORTED: "+
                    std::string(name)+" (raw zero must represent signed-int8 zero)");
        }
        const auto& bytes=prefill.input_buffers.at(i);
        prefill_input_bytes+=bytes.size();
        prefill_input_nonzero+=static_cast<size_t>(std::count_if(
            bytes.begin(),bytes.end(),[](uint8_t b){return b!=0;}));
    }
    if(prefill_input_kv!=28)
        throw std::runtime_error("KV_PREFILL_PROBE_EXPECTED_28_PREFILL_KV_INPUTS");
    if(linked_caches && prefill_input_nonzero!=0)
        throw std::runtime_error(
            "PREFILL_LINK_REQUIRES_ZERO_INITIAL_28_KV: saved Prefill sample "
            "has nonzero initial K/V; this is not a proven fresh prompt");
    std::cout<<"KV_PREFILL_SOURCE_CACHE_INFO: kv_inputs="<<prefill_input_kv
             <<" bytes="<<prefill_input_bytes
             <<" nonzero_bytes="<<prefill_input_nonzero
             <<" scratch_decode_cache_zero_initialized=1\n"<<std::flush;

    std::set<size_t> seen;
    size_t inserted=0;
    uint64_t scratch_bytes=0;
    for(size_t n=0;n<links.size();++n) {
        const auto& link=links[n];
        const auto expected_decode_output=
            "StatefulPartitionedCall_"+std::to_string(n);
        const char* decode_out_name=tensor_name(
            decode.outputs.at(link.output_index));
        if(!decode_out_name || expected_decode_output!=decode_out_name)
            throw std::runtime_error(
                "KV_PREFILL_PROBE_DECODE_ORDER_UNEXPECTED: "+link.name);

        const std::string source_name=
            "StatefulPartitionedCall_1_"+std::to_string(n);
        size_t from=prefill.outputs.size();
        for(size_t j=0;j<prefill.outputs.size();++j) {
            const char* candidate=tensor_name(prefill.outputs[j]);
            if(candidate && source_name==candidate) {
                if(from!=prefill.outputs.size())
                    throw std::runtime_error(
                        "KV_PREFILL_PROBE_DUPLICATE_OUTPUT: "+source_name);
                from=j;
            }
        }
        if(from==prefill.outputs.size() || !seen.insert(from).second)
            throw std::runtime_error("KV_PREFILL_PROBE_OUTPUT_NOT_UNIQUE: "+source_name);

        const auto& source=prefill.outputs[from];
        const auto& target=decode.inputs.at(link.input_index);
        const auto* sh=tensor_dims(source);
        if(tensor_rank(source)!=4 || tensor_dtype(source)!=QNN_DATATYPE_SFIXED_POINT_8 ||
           sh[0]!=1 || sh[1]!=1 ||
           (link.key ? sh[2]!=link.channels || sh[3]!=kBlockRows :
                       sh[2]!=kBlockRows || sh[3]!=link.channels) ||
           tensor_nbytes(source)!=static_cast<size_t>(link.channels)*kBlockRows)
            throw std::runtime_error(
                "KV_PREFILL_PROBE_SOURCE_SHAPE_MISMATCH: "+source_name);
        const auto& qsrc=tensor_quant(source);
        const auto& qdst=tensor_quant(target);
        if(linked_caches &&
           qdst.quantizationEncoding==QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
           qdst.scaleOffsetEncoding.offset!=0)
            throw std::runtime_error(
                "PREFILL_LINK_NONZERO_QUANT_ZERO_POINT: scratch byte zero is "
                "not numerical zero for "+link.name);
        if(qsrc.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           qdst.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
            throw std::runtime_error(
                "KV_PREFILL_PROBE_QUANT_ENCODING_UNSUPPORTED: "+source_name);
        if(qsrc.scaleOffsetEncoding.scale!=qdst.scaleOffsetEncoding.scale ||
           qsrc.scaleOffsetEncoding.offset!=qdst.scaleOffsetEncoding.offset) {
            std::ostringstream msg;
            msg<<"KV_PREFILL_PROBE_QUANT_MISMATCH: "<<source_name
               <<" -> "<<link.name
               <<" src_scale="<<qsrc.scaleOffsetEncoding.scale
               <<" dst_scale="<<qdst.scaleOffsetEncoding.scale
               <<" src_offset="<<qsrc.scaleOffsetEncoding.offset
               <<" dst_offset="<<qdst.scaleOffsetEncoding.offset;
            throw std::runtime_error(msg.str());
        }
        const auto& block=prefill.output_buffers.at(from);
        if(block.size()!=tensor_nbytes(source))
            throw std::runtime_error(
                "KV_PREFILL_PROBE_SOURCE_OUTPUT_BUFFER_INVALID: "+source_name);
        // Never start from the independent saved Decode sample. A future
        // true prompt path must initialize its own 30 K/V caches correctly.
        std::vector<uint8_t> scratch(tensor_nbytes(target),0);
        if(copied_rows)
            gemma4::update_kv_block(
                scratch,block,link.channels,kCapacity,0,kBlockRows,copied_rows,
                link.key?gemma4::KvLayout::Key:gemma4::KvLayout::Value);
        // Independent readback of all inserted bytes (not just a checksum).
        for(size_t row=0;row<copied_rows;++row) {
            for(size_t c=0;c<link.channels;++c) {
                const size_t src_offset=link.key?
                    c*kBlockRows+row:row*link.channels+c;
                const size_t dst_offset=link.key?
                    c*kCapacity+row:row*link.channels+c;
                if(scratch[dst_offset]!=block[src_offset])
                    throw std::runtime_error(
                        "KV_PREFILL_PROBE_KV_COPY_MISMATCH: "+source_name);
            }
        }
        // Because scratch starts entirely zero, every unused/padded cache
        // slot must stay zero. No accidental padding writes permitted.
        size_t nonzero_tail=0;
        for(size_t row=copied_rows;row<kCapacity;++row) {
            for(size_t c=0;c<link.channels;++c) {
                const size_t offset=link.key?
                    c*kCapacity+row:row*link.channels+c;
                if(scratch[offset]!=0)++nonzero_tail;
            }
        }
        if(nonzero_tail)
            throw std::runtime_error(
                "KV_PREFILL_PROBE_PADDING_WRITTEN: "+source_name);
        scratch_bytes+=scratch.size();
        inserted+=static_cast<size_t>(link.channels)*copied_rows;
        if(linked_caches)(*linked_caches)[link.input_index]=std::move(scratch);
        std::cout<<"KV_PREFILL_PROBE_LINK_OK: output="<<source_name
                 <<" decode_input="<<link.name
                 <<" type="<<(link.key?"K":"V")
                 <<" channels="<<link.channels
                 <<" rows_copied="<<copied_rows
                 <<" padding_rows_skipped="<<(kBlockRows-valid_tokens)
                 <<" last_real_token_held_out="<<(seed_last_prompt?1:0)
                 <<" first_prompt_replay="<<(seed_first_prompt?1:0)
                 <<" output_index="<<from
                 <<" decode_index="<<link.input_index
                 <<" quant_scale="<<qsrc.scaleOffsetEncoding.scale
                 <<"\n"<<std::flush;
    }
    if(seen.size()!=30)
        throw std::runtime_error("KV_PREFILL_PROBE_NOT_30_UNIQUE_OUTPUTS");
    if(linked_caches) {
        for(const auto& link:links)
            if(linked_caches->at(link.input_index).size() !=
               tensor_nbytes(decode.inputs.at(link.input_index)))
                throw std::runtime_error("PREFILL_LINK_KV_BUFFER_INCOMPLETE");
    }
    std::cout<<"KV_PREFILL_BRIDGE_PROBE_OK: tensors=30 layers=15"
             <<" valid_tokens="<<valid_tokens
             <<" padding_rows_ignored="<<(kBlockRows-valid_tokens)
             <<" copied_rows="<<copied_rows
             <<" next_decode_position="<<copied_rows
             <<" last_prompt_replay="<<(seed_last_prompt?1:0)
             <<" first_prompt_replay="<<(seed_first_prompt?1:0)
             <<" copied_bytes="<<inserted
             <<" scratch_cache_bytes="<<scratch_bytes
             <<" prefill_kv_input_nonzero_bytes="<<prefill_input_nonzero
             <<" quantization_match=1 no_active_decode_mutation=1"
             <<" NPU_decode_link_not_executed=1\n"<<std::flush;
}

// Stage 6.6: one experimental Decode step using exactly 30 K/V scratch
// buffers produced by Prefill-128, plus all 8 non-KV tensors recomputed for
// token 568 (or another explicit probe ID) at position valid_tokens.
//
// CRITICAL: This does not reuse a single saved Decode input byte. The 28
// original Prefill KV inputs must be all zero, and their QNN quantized-zero
// offsets must be compatible (enforced in the mapping probe). It does not
// compare linked Decode outputs to the independent saved Decode references.
#include "stage6_12_bos_only.inc"
#include "stage6_15_kv_histories.inc"
#include "stage6_14_decode_only.inc"

void execute_prefill_linked_decode(
    QnnRuntime& runtime,const ReplayJob& prefill,ReplayJob& decode,
    ReplayJob& head,size_t hidden_index,
    const Options& options,
    std::vector<std::vector<uint8_t>> scratch_caches) {
    constexpr size_t kPositions=128;
    const auto bytes=read_file(options.cpu_prefill_tokens);
    if(bytes.size()!=kPositions*sizeof(int32_t))
        throw std::runtime_error("PREFILL_LINK_TOKEN_BYTES_INVALID");
    std::array<int32_t,kPositions> prompt{};
    std::memcpy(prompt.data(),bytes.data(),bytes.size());
    size_t first_pad=kPositions;
    for(size_t i=0;i<kPositions;++i) {
        if(prompt[i]==0 && first_pad==kPositions)first_pad=i;
        if(prompt[i]!=0 && first_pad!=kPositions)
            throw std::runtime_error("PREFILL_LINK_NONCONTIGUOUS_PADDING");
    }
    if(first_pad==0)
        throw std::runtime_error("PREFILL_LINK_EMPTY_PROMPT");
    const size_t first_position=options.prefill_seed_first_prompt?0:
        first_pad-(options.prefill_seed_last_prompt?1:0);
    const int32_t first_token=options.prefill_seed_first_prompt?prompt[0]:
        (options.prefill_seed_last_prompt?
         prompt[first_pad-1]:options.prefill_kv_token);
    if(first_position>=4096 || first_token<0 || first_token>=262144)
        throw std::runtime_error("PREFILL_LINK_POSITION_OR_TOKEN_INVALID");
    if(options.prefill_seed_first_prompt) {
        if(prompt[0]==0)
            throw std::runtime_error("PREFILL_FIRST_PROMPT_ID_IS_PADDING");
        std::cout<<"PREFILL_FIRST_PROMPT_SEED: prompt_tokens="<<first_pad
                 <<" replay_position=0"
                 <<" replay_token="<<first_token
                 <<" kv_rows_from_prefill=0"
                 <<" all_KV_raw_zero=1 no_previous_attention_context=1\n"
                 <<std::flush;
    }
    if(options.prefill_seed_last_prompt) {
        if(first_pad<2 || prompt[first_pad-1]==0)
            throw std::runtime_error("PREFILL_LAST_PROMPT_TOKEN_INVALID");
        std::cout<<"PREFILL_LAST_PROMPT_SEED: prompt_tokens="<<first_pad
                 <<" replay_position="<<first_position
                 <<" replay_token="<<first_token
                 <<" kv_rows_from_prefill="<<first_position
                 <<" first_generated_token_from_LM_Head=1\n"<<std::flush;
    }
    if(prefill.outputs.size()!=30 || scratch_caches.size()!=decode.inputs.size() ||
       decode.outputs.size()!=31 || head.outputs.size()!=1)
        throw std::runtime_error("PREFILL_LINK_GRAPH_METADATA_MISMATCH");
    const auto kv=decode_kv_links(decode);
    std::set<size_t> cache_indices;
    for(const auto& link:kv) {
        if(!cache_indices.insert(link.input_index).second ||
           scratch_caches[link.input_index].size()!=
               tensor_nbytes(decode.inputs[link.input_index]))
            throw std::runtime_error("PREFILL_LINK_KV_INCOMPLETE");
    }

    // Both the embedding and Section1 RoPE/mask paths are direct CPU calls,
    // deliberately without per-step reference samples. Every tensor is
    // validated by original QNN metadata before the graph is executed.
    std::set<size_t> nonkv_indices;
    const auto assign=[&](const std::string& name,Qnn_DataType_t dtype,
                          size_t expected,std::vector<uint8_t> data) {
        size_t index=decode.inputs.size();
        for(size_t i=0;i<decode.inputs.size();++i) {
            const char* actual=tensor_name(decode.inputs[i]);
            if(actual && name==actual) {
                if(index!=decode.inputs.size())
                    throw std::runtime_error("PREFILL_LINK_DUPLICATE_NAME: "+name);
                index=i;
            }
        }
        if(index==decode.inputs.size() || cache_indices.count(index) ||
           !nonkv_indices.insert(index).second ||
           tensor_dtype(decode.inputs[index])!=dtype ||
           tensor_nbytes(decode.inputs[index])!=expected ||
           data.size()!=expected)
            throw std::runtime_error("PREFILL_LINK_NONKV_METADATA_INVALID: "+name);
        scratch_caches[index]=std::move(data);
        std::cout<<"PREFILL_LINK_NONKV_READY: name="<<name
                 <<" qnn_index="<<index
                 <<" bytes="<<expected<<"\n"<<std::flush;
    };
    assign("decode_embeddings_0",QNN_DATATYPE_FLOAT_32,6144,
        gemma4::compute_packed_embedding(
            options.cpu_embed_section0,0,first_token));
    assign("decode_per_layer_embeddings_0",QNN_DATATYPE_FLOAT_32,35840,
        gemma4::compute_packed_embedding(
            options.cpu_embed_section2,2,first_token));
    auto aux=gemma4::run_decode_aux(options.cpu_aux_model,
        static_cast<int32_t>(first_position),first_token);
    if(aux.size()!=6)
        throw std::runtime_error("PREFILL_LINK_REQUIRES_SIX_SECTION1_TENSORS");
    const std::array<size_t,6> expected_lengths{512,512,256,256,4097,4097};
    // The original Section1 helper yields RoPE 4 + mask 2 in this order.
    for(size_t j=0;j<aux.size();++j)
        assign(aux[j].qnn_name,QNN_DATATYPE_SFIXED_POINT_8,
               expected_lengths[j],std::move(aux[j].bytes));
    if(cache_indices.size()!=30 || nonkv_indices.size()!=8)
        throw std::runtime_error("PREFILL_LINK_REQUIRES_EXACTLY_38_INPUTS");
    std::vector<Qnn_Tensor_t> linked_inputs=decode.inputs;
    for(size_t i=0;i<linked_inputs.size();++i) {
        if(scratch_caches[i].size()!=tensor_nbytes(linked_inputs[i]))
            throw std::runtime_error("PREFILL_LINK_RAW_INPUT_INCOMPLETE: "+
                                     std::to_string(i));
        bind_raw(linked_inputs[i],scratch_caches[i]);
    }
    std::cout<<"PREFILL_LINK_INPUTS_READY: kv_from_prefill_ram=30"
             <<" nonkv_from_cpu=8"
             <<" position="<<first_position
             <<" input_token="<<first_token
             <<" saved_decode_inputs_used=0"
             <<" fresh_prefill_initial_kv_zero=1\n"<<std::flush;

    const auto start=std::chrono::steady_clock::now();
    runtime.execute(decode.handle,linked_inputs,decode.outputs);
    const auto stop=std::chrono::steady_clock::now();
    // All 31 new output buffers now belong to the linked Decode. They
    // must never be compared against independent saved Decode samples.
    std::cout<<"PREFILL_LINK_DECODE_OK: graph="<<decode.graph_name
             <<" outputs="<<decode.outputs.size()
             <<" position="<<first_position
             <<" input_token="<<first_token
             <<" execute_ms="
             <<std::chrono::duration<double,std::milli>(stop-start).count()
             <<" independent_decode_reference_comparison=0\n"<<std::flush;
    if(options.prefill_seed_last_prompt || options.prefill_seed_first_prompt) {
        // Independently compare all 30 replayed-token Decode KV slices with the
        // last row of the Prefill-128 output blocks. This is an extra
        // cross-graph numerical diagnostic; do not assume bit-exactness
        // between distinct QNN kernels. Record differences honestly.
        constexpr size_t kPrefillRows=128;
        size_t total=0, different=0, exact_tensors=0;
        size_t diagnostic_best_row_16=0, diagnostic_best_source_expected=0;
        size_t diagnostic_exact_alternate_row=0, diagnostic_exact_alternate_source=0;
        std::set<size_t> seen_source;
        for(size_t n=0;n<kv.size();++n) {
            const auto& link=kv[n];
            const std::string source_name=
                "StatefulPartitionedCall_1_"+std::to_string(n);
            size_t index=prefill.outputs.size();
            for(size_t j=0;j<prefill.outputs.size();++j) {
                const char* actual=tensor_name(prefill.outputs[j]);
                if(actual && source_name==actual) {
                    if(index!=prefill.outputs.size())
                        throw std::runtime_error(
                            "PREFILL_LAST_KV_DUPLICATE_SOURCE: "+source_name);
                    index=j;
                }
            }
            if(index==prefill.outputs.size() ||
               !seen_source.insert(index).second)
                throw std::runtime_error(
                    "PREFILL_LAST_KV_SOURCE_MISSING: "+source_name);
            const auto& source=prefill.output_buffers[index];
            const auto& actual=decode.output_buffers[link.output_index];
            if(source.size()!=static_cast<size_t>(link.channels)*kPrefillRows ||
               actual.size()!=link.channels)
                throw std::runtime_error(
                    "PREFILL_LAST_KV_OUTPUT_SIZE_MISMATCH: "+source_name);
            size_t mismatch=0;
            for(size_t c=0;c<link.channels;++c) {
                const size_t src_index=link.key?
                    c*kPrefillRows+first_position:
                    first_position*link.channels+c;
                if(actual[c]!=source[src_index])++mismatch;
            }
            total+=link.channels;
            different+=mismatch;
            if(mismatch==0)++exact_tensors;
            std::cout<<(options.prefill_seed_first_prompt?
                         "PREFILL_FIRST_PROMPT_KV_SLICE_COMPARE: ":
                         "PREFILL_LAST_PROMPT_KV_SLICE_COMPARE: ")
                     <<"name="<<source_name
                     <<" kind="<<(link.key?"K":"V")
                     <<" bytes="<<link.channels
                     <<" mismatching_bytes="<<mismatch<<"\n"<<std::flush;
            if(options.prefill_kv_diagnose) {
                // Diagnose, but NEVER correct, last-token Prefill/Decode KV
                // incompatibility. Find whether a shifted Prefill time row
                // or a differently named layer yields a closer match.
                // Candidates must preserve K vs V layout, channel count,
                // dtype and identical QNN scale/offset quantization.
                const auto signed_q=[](uint8_t b) -> int {
                    return b<128?static_cast<int>(b):static_cast<int>(b)-256;
                };
                size_t abs1=0,abs2=0,abs_large=0,src_zero=0,dst_zero=0;
                size_t abs_sum=0,abs_max=0;
                for(size_t c=0;c<link.channels;++c) {
                    const size_t off=link.key?
                        c*kPrefillRows+first_position:
                        first_position*link.channels+c;
                    const int x=signed_q(actual[c]),y=signed_q(source[off]);
                    const size_t d=static_cast<size_t>(std::abs(x-y));
                    abs_sum+=d;
                    abs_max=std::max(abs_max,d);
                    if(d==1)++abs1;
                    else if(d==2)++abs2;
                    else if(d>2)++abs_large;
                    if(y==0)++src_zero;
                    if(x==0)++dst_zero;
                }
                const auto count_row_diff=[&](const std::vector<uint8_t>& block,
                                              size_t row) -> size_t {
                    size_t count=0;
                    for(size_t c=0;c<link.channels;++c) {
                        const size_t off=link.key?
                            c*kPrefillRows+row:row*link.channels+c;
                        count+=static_cast<size_t>(actual[c]!=block[off]);
                    }
                    return count;
                };
                size_t best_row=kPrefillRows;
                size_t best_row_diff=link.channels+1;
                for(size_t row=0;row<kPrefillRows;++row) {
                    const size_t d=count_row_diff(source,row);
                    if(d<best_row_diff) {
                        best_row_diff=d;
                        best_row=row;
                    }
                }
                if(best_row==first_position)++diagnostic_best_row_16;
                if(best_row_diff==0)++diagnostic_exact_alternate_row;
                // Stage 6.5 already proved Prefill source quantization matches
                // the Decode *input cache*. Do not assume an independent
                // QNN slice output encoding when screening layer candidates.
                const auto& expected_quant=tensor_quant(decode.inputs.at(link.input_index));
                std::string best_source_name="(none)";
                size_t best_source_diff=link.channels+1;
                size_t compatible=0;
                for(size_t candidate=0;candidate<prefill.outputs.size();++candidate) {
                    const auto& tensor=prefill.outputs[candidate];
                    if(tensor_dtype(tensor)!=QNN_DATATYPE_SFIXED_POINT_8 ||
                       tensor_rank(tensor)!=4)continue;
                    const auto* d=tensor_dims(tensor);
                    if(d[0]!=1 || d[1]!=1 ||
                       (link.key?(d[2]!=link.channels || d[3]!=kPrefillRows):
                                 (d[2]!=kPrefillRows || d[3]!=link.channels)))
                        continue;
                    const auto& quant=tensor_quant(tensor);
                    if(quant.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
                       expected_quant.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
                       quant.scaleOffsetEncoding.scale!=expected_quant.scaleOffsetEncoding.scale ||
                       quant.scaleOffsetEncoding.offset!=expected_quant.scaleOffsetEncoding.offset)
                        continue;
                    const auto& block=prefill.output_buffers[candidate];
                    if(block.size()!=source.size())
                        throw std::runtime_error("PREFILL_KV_DIAG_CANDIDATE_SIZE_INVALID");
                    ++compatible;
                    const size_t dcount=count_row_diff(block,first_position);
                    if(dcount<best_source_diff) {
                        best_source_diff=dcount;
                        const char* nm=tensor_name(tensor);
                        best_source_name=nm?nm:"(unnamed)";
                    }
                }
                if(best_source_name==source_name)++diagnostic_best_source_expected;
                if(best_source_diff==0)++diagnostic_exact_alternate_source;
                if(compatible==0 || best_source_diff>link.channels ||
                   best_row>=kPrefillRows)
                    throw std::runtime_error("PREFILL_KV_DIAG_NO_VALID_SOURCE");
                std::cout<<"PREFILL_KV_DIAG_TENSOR: mapped_output="<<source_name
                         <<" decode_cache="<<link.name
                         <<" kind="<<(link.key?"K":"V")
                         <<" channels="<<link.channels
                         <<" expected_row="<<first_position
                         <<" mismatched_bytes="<<mismatch
                         <<" abs_delta1="<<abs1
                         <<" abs_delta2="<<abs2
                         <<" abs_delta_gt2="<<abs_large
                         <<" sum_abs_delta="<<abs_sum
                         <<" max_abs_delta="<<abs_max
                         <<" src_signed_zero="<<src_zero
                         <<" decode_signed_zero="<<dst_zero
                         <<" best_same_tensor_row="<<best_row
                         <<" best_same_tensor_row_mismatches="<<best_row_diff
                         <<" compatible_layer_candidates="<<compatible
                         <<" best_compatible_source="<<best_source_name
                         <<" best_compatible_source_mismatches="<<best_source_diff
                         <<" read_only=1\n"<<std::flush;
            }
        }
        if(options.prefill_kv_diagnose) {
            std::cout<<"PREFILL_KV_DIAG_SUMMARY: tensors="<<kv.size()
                     <<" compared_bytes="<<total
                     <<" mismatched_bytes="<<different
                     <<" best_same_tensor_row_is_expected="
                     <<diagnostic_best_row_16
                     <<" best_compatible_layer_is_expected="
                     <<diagnostic_best_source_expected
                     <<" exact_some_same_tensor_row="
                     <<diagnostic_exact_alternate_row
                     <<" exact_some_compatible_layer="
                     <<diagnostic_exact_alternate_source
                     <<" no_relabel_or_patch_applied=1"
                     <<" diagnostic_only=1\n"<<std::flush;
        }
        std::cout<<(options.prefill_seed_first_prompt?
                     "PREFILL_FIRST_PROMPT_KV_COMPARE: tensors=30":
                     "PREFILL_LAST_PROMPT_KV_COMPARE: tensors=30")
                 <<" compared_bytes="<<total
                 <<" mismatching_bytes="<<different
                 <<" exact_tensors="<<exact_tensors
                 <<" bit_exact="<<(different==0?1:0)
                 <<" position="<<first_position
                 <<" numerical_equivalence_not_assumed=1\n"<<std::flush;
    }
    if(options.prefill_bos_only_compare)
        run_stage6_12_bos_only(runtime,prefill,decode,kv,options,scratch_caches);
    execute_chained_decode_head(runtime,decode,head,hidden_index,true);
    std::cout<<"PREFILL_LINK_HEAD_OK: logits_bytes="<<head.output_buffers[0].size()
             <<" linked_prefill_kv=1"
             <<" no_saved_decode_reference_claim=1\n"<<std::flush;

    // Stage 6.7: deterministic, reference-free continuation of the exact
    // Prefill-linked Decode state. The *first* input is still an explicit
    // probe ID. Only subsequent IDs come from quantized LM Head argmax.
    // Keep all KV in the same scratch RAM and never restore saved samples.
    const auto next_argmax=[&](uint32_t executed_step) -> int32_t {
        const auto& logits=head.output_buffers.at(0);
        if(tensor_dtype(head.outputs.at(0))!=QNN_DATATYPE_SFIXED_POINT_8 ||
           logits.size()!=262144)
            throw std::runtime_error("PREFILL_LINK_GREEDY_LOGITS_METADATA_INVALID");
        int best=-129;
        int32_t id=0;
        uint32_t ties=0;
        for(size_t i=0;i<logits.size();++i) {
            const int q=logits[i]<128?static_cast<int>(logits[i]):
                                          static_cast<int>(logits[i])-256;
            if(q>best) {
                best=q;
                id=static_cast<int32_t>(i);
                ties=1;
            } else if(q==best) ++ties;
        }
        // The QNN LM Head emits only quantized INT8 logits, not FLOAT32.
        // Expose every ID sharing the maximum instead of pretending that
        // the first ID was uniquely better. The scan keeps low-ID ordering.
        std::vector<int32_t> tied_ids;
        tied_ids.reserve(ties);
        for(size_t i=0;i<logits.size();++i) {
            const int q=logits[i]<128?static_cast<int>(logits[i]):
                                          static_cast<int>(logits[i])-256;
            if(q==best)tied_ids.push_back(static_cast<int32_t>(i));
        }
        if(tied_ids.size()!=ties || tied_ids.empty() || tied_ids.front()!=id)
            throw std::runtime_error("PREFILL_LINK_TIE_DIAGNOSTIC_INCONSISTENT");
        std::ostringstream ids;
        for(size_t i=0;i<tied_ids.size();++i)
            ids<<(i?",":"")<<tied_ids[i];
        std::cout<<"PREFILL_LINK_GREEDY_CANDIDATES: executed_step="
                 <<executed_step<<" max_logit_q="<<best
                 <<" ties="<<ties
                 <<" candidate_token_ids="<<ids.str()<<"\n"<<std::flush;
        if(ties!=1) {
            if(!options.prefill_allow_tie_lowest)
                throw std::runtime_error(
                    "PREFILL_LINK_GREEDY_AMBIGUOUS_INT8_TIE: "
                    "use --prefill-tie-policy lowest-id only for experiments; "
                    "a quantized tie does not determine the true FP32 winner");
            std::cout<<"PREFILL_LINK_TIE_RESOLUTION: executed_step="
                     <<executed_step
                     <<" policy=lowest-id chosen_token_id="<<id
                     <<" tied_ids="<<ids.str()
                     <<" experimental_arbitrary_tiebreak=1"
                     <<" no_FP32_ordering_claim=1\n"<<std::flush;
        }
        std::cout<<"PREFILL_LINK_GREEDY_NEXT: executed_step="<<executed_step
                 <<" token_id="<<id<<" max_logit_q="<<best
                 <<" argmax_ties="<<ties
                 <<" tie_policy="<<(ties==1?"unique":"lowest-id-experimental")
                 <<" quantized_int8_argmax=1\n"<<std::flush;
        return id;
    };
    if(options.prefill_kv_steps<1 || options.prefill_kv_steps>16)
        throw std::runtime_error("PREFILL_LINK_STEPS_OUT_OF_RANGE");
    if(first_position+options.prefill_kv_steps>4096)
        throw std::runtime_error("PREFILL_LINK_STEPS_EXCEED_KV_CAPACITY");
    uint32_t executed_steps=1;
    bool eos_stop=false;
    const auto stop_if_eos=[&](int32_t id,uint32_t prediction_step)->bool {
        if(!options.prefill_stop_on_eos || (id!=1 && id!=106 && id!=50))
            return false;
        std::cout<<"PREFILL_LINK_STOP_EOS: prediction_step="<<prediction_step
                 <<" predicted_position="<<(first_position+prediction_step)
                 <<" eos_id="<<id
                 <<" not_submitted_to_decode=1"
                 <<" generation_complete=1\n"<<std::flush;
        return true;
    };
    int32_t next_id=first_token;
    if(options.prefill_seed_last_prompt) {
        next_id=next_argmax(1);
        std::cout<<"PREFILL_FIRST_TOKEN_FROM_PROMPT: token_id="<<next_id
                 <<" prompt_tokens="<<first_pad
                 <<" replayed_last_prompt_position="<<first_position
                 <<" quantized_greedy=1 independent_reference_claim=0\n"
                 <<std::flush;
    } else if(options.prefill_kv_steps>1) {
        next_id=next_argmax(1);
        eos_stop=stop_if_eos(next_id,1);
    }
    for(uint32_t step=2;step<=options.prefill_kv_steps && !eos_stop;++step) {
        const size_t feedback_position=first_position+step-2;
        const size_t input_position=feedback_position+1;
        size_t written=0;
        for(const auto& link:kv) {
            const auto& slice=decode.output_buffers.at(link.output_index);
            auto& cache=scratch_caches.at(link.input_index);
            if(slice.size()!=link.channels ||
               cache.size()!=tensor_nbytes(decode.inputs.at(link.input_index)))
                throw std::runtime_error(
                    "PREFILL_LINK_FEEDBACK_METADATA_MISMATCH: "+link.name);
            gemma4::update_kv_slot(
                cache,slice,link.channels,4096,feedback_position,
                link.key?gemma4::KvLayout::Key:gemma4::KvLayout::Value);
            // Independently verify all inserted bytes in the 30 KV caches.
            for(size_t c=0;c<link.channels;++c) {
                const size_t dst=link.key?
                    c*4096+feedback_position:feedback_position*link.channels+c;
                if(cache[dst]!=slice[c])
                    throw std::runtime_error(
                        "PREFILL_LINK_FEEDBACK_VERIFY_FAILED: "+link.name);
            }
            written+=slice.size();
        }
        if(written!=9216)
            throw std::runtime_error("PREFILL_LINK_FEEDBACK_EXPECTED_9216_BYTES");
        std::cout<<"PREFILL_LINK_KV_FEEDBACK_OK: executed_step="<<(step-1)
                 <<" position="<<feedback_position
                 <<" cache_tensors="<<kv.size()
                 <<" inserted_bytes="<<written
                 <<" layout=K_strided,V_contiguous\n"<<std::flush;

        // Re-create all eight CPU non-KV tensors for the new position and
        // token, without reading the saved Decode input list.
        nonkv_indices.clear();
        assign("decode_embeddings_0",QNN_DATATYPE_FLOAT_32,6144,
            gemma4::compute_packed_embedding(
                options.cpu_embed_section0,0,next_id));
        assign("decode_per_layer_embeddings_0",QNN_DATATYPE_FLOAT_32,35840,
            gemma4::compute_packed_embedding(
                options.cpu_embed_section2,2,next_id));
        auto next_aux=gemma4::run_decode_aux(
            options.cpu_aux_model,static_cast<int32_t>(input_position),next_id);
        if(next_aux.size()!=6)
            throw std::runtime_error(
                "PREFILL_LINK_NEXT_AUX_COUNT_NOT_SIX");
        for(size_t j=0;j<next_aux.size();++j)
            assign(next_aux[j].qnn_name,QNN_DATATYPE_SFIXED_POINT_8,
                   expected_lengths[j],std::move(next_aux[j].bytes));
        if(nonkv_indices.size()!=8)
            throw std::runtime_error("PREFILL_LINK_NEXT_NONKV_COUNT_NOT_EIGHT");
        // Rebinding is required: changing std::vector storage invalidates
        // the QNN raw client buffer pointer held by linked_inputs.
        for(size_t i=0;i<linked_inputs.size();++i) {
            if(scratch_caches[i].size()!=tensor_nbytes(linked_inputs[i]))
                throw std::runtime_error(
                    "PREFILL_LINK_NEXT_INPUT_SIZE_CHANGED: "+std::to_string(i));
            bind_raw(linked_inputs[i],scratch_caches[i]);
        }
        std::cout<<"PREFILL_LINK_NEXT_INPUTS_READY: step="<<step
                 <<" position="<<input_position
                 <<" token="<<next_id
                 <<" kv_from_feedback_ram=30"
                 <<" nonkv_from_cpu=8"
                 <<" saved_decode_inputs_used=0\n"<<std::flush;
        const auto decode_start=std::chrono::steady_clock::now();
        runtime.execute(decode.handle,linked_inputs,decode.outputs);
        const auto decode_end=std::chrono::steady_clock::now();
        std::cout<<"PREFILL_LINK_NEXT_DECODE_OK: step="<<step
                 <<" position="<<input_position
                 <<" input_token="<<next_id
                 <<" outputs=31"
                 <<" decode_ms="
                 <<std::chrono::duration<double,std::milli>(
                     decode_end-decode_start).count()
                 <<" reference_comparison=0\n"<<std::flush;
        execute_chained_decode_head(runtime,decode,head,hidden_index,true);
        executed_steps=step;
        std::cout<<"PREFILL_LINK_NEXT_HEAD_OK: step="<<step
                 <<" logits_bytes="<<head.output_buffers.at(0).size()
                 <<" linked_KV_feedback=1"
                 <<" independent_reference_comparison=0\n"<<std::flush;
        if(options.prefill_stop_on_eos || step<options.prefill_kv_steps) {
            next_id=next_argmax(step);
            eos_stop=stop_if_eos(next_id,step);
        }
    }
    if(options.prefill_stop_on_eos && !eos_stop)
        std::cout<<"PREFILL_LINK_STOP_LIMIT: executed_steps="<<executed_steps
                 <<" requested_steps="<<options.prefill_kv_steps
                 <<" generation_complete=0\n"<<std::flush;
    std::cout<<"PREFILL_LINK_MULTI_STEP_OK: steps="<<executed_steps
             <<" first_position="<<first_position
             <<" last_position="<<(first_position+executed_steps-1)
             <<" first_token="<<first_token
             <<" last_prompt_replayed="<<(options.prefill_seed_last_prompt?1:0)
             <<" first_prompt_replayed="<<(options.prefill_seed_first_prompt?1:0)
             <<" int8_tie_policy="<<(options.prefill_allow_tie_lowest?"lowest-id-experimental":"stop")
             <<" requested_steps="<<options.prefill_kv_steps
             <<" eos_stop="<<(eos_stop?1:0)
             <<" all_KV_from_prefill_or_feedback_RAM=1"
             <<" all_nonKV_from_CPU=1"
             <<" independent_reference_claim=0\n"<<std::flush;
    if(options.decode_only_prompt)
        run_stage6_14_decode_only(runtime,prefill,decode,head,hidden_index,
                                  kv,options,scratch_caches);
}

int run_replay_jobs(const Options& options) {
    auto jobs = parse_replay_jobs(options.jobs);
    // Check all files, all tensor sizes and graph metadata BEFORE NPU init.
    for (auto& job : jobs) prepare_replay_job(*job, options.system);

    ReplayJob* decode = nullptr;
    ReplayJob* head = nullptr;
    size_t hidden_output_index = 0;
    if (options.chain_decode_head) {
        // For this Q8B smoke test, use the already verified Decode-first load order.
        // Prefill-first previously failed mapping the large Decode weights buffer.
        if (jobs.front()->graph_name != "qnn_partition_0")
            throw std::runtime_error("--chain-decode-head requires a Decode-first jobs manifest on Q8B");
        for (auto& job : jobs) {
            if (job->graph_name == "qnn_partition_0") {
                if (decode) throw std::runtime_error("Duplicate Decode job in --jobs");
                decode = job.get();
            } else if (job->graph_name == "qnn_partition_1") {
                if (head) throw std::runtime_error("Duplicate LM Head job in --jobs");
                head = job.get();
            }
        }
        if (!decode || !head)
            throw std::runtime_error("--chain-decode-head requires Decode + LM Head jobs");
        hidden_output_index = find_decode_head_link(*decode, *head);
        std::cout << "CHAIN_METADATA_OK: decode_output_index=" << hidden_output_index
                  << " hidden_bytes=" << tensor_nbytes(decode->outputs[hidden_output_index])
                  << " head_input_bytes=" << tensor_nbytes(head->inputs[0])
                  << "\n" << std::flush;
    }

    std::vector<KvLink> kv_links;
    std::vector<fs::path> next_cache_files;
    std::vector<fs::path> next_decode_references;
    fs::path next_head_reference;
    if (!options.kv_next_head_reference_dir.empty()) {
        if (!head || head->outputs.size() != 1)
            throw std::runtime_error("Next LM Head reference requires a valid LM Head job");
        const fs::path dir(options.kv_next_head_reference_dir);
        next_head_reference = dir / native_basename(head->outputs[0], 0);
        if (!fs::is_regular_file(next_head_reference) ||
            fs::file_size(next_head_reference) !=
                tensor_nbytes(head->outputs[0]))
            throw std::runtime_error("Invalid next LM Head reference: " +
                                     next_head_reference.string());
        std::cout << "DECODE_NEXT_HEAD_REFERENCE_READY: "
                  << next_head_reference.string() << "\n" << std::flush;
    }
    if (options.kv_feedback) {
        if (jobs.front()->graph_name != "qnn_partition_0")
            throw std::runtime_error("--kv-feedback-pos requires a Decode-first manifest");
        for (auto& job : jobs) {
            if (job->graph_name != "qnn_partition_0") continue;
            if (decode && decode != job.get())
                throw std::runtime_error("Duplicate Decode graph in jobs");
            decode = job.get();
        }
        if (!decode) throw std::runtime_error("KV feedback needs Decode");
        kv_links = decode_kv_links(*decode);
        if (!options.kv_next_list.empty()) {
            const auto& graph = decode->metadata->find(decode->graph_name);
            next_cache_files = parse_qnn_input_list(options.kv_next_list, graph);
            for (const auto& link : kv_links) {
                const auto& p = next_cache_files.at(link.input_index);
                if (!fs::is_regular_file(p) ||
                    fs::file_size(p) != tensor_nbytes(decode->inputs[link.input_index]))
                    throw std::runtime_error("Invalid next-sample KV file: " + p.string());
            }
            if (options.kv_run_next) {
                // Validate all 38 next inputs before initializing the NPU.
                for (size_t i = 0; i < decode->inputs.size(); ++i) {
                    const auto& p = next_cache_files.at(i);
                    if (!fs::is_regular_file(p) ||
                        fs::file_size(p) != tensor_nbytes(decode->inputs[i]))
                        throw std::runtime_error("Invalid next Decode input: " + p.string());
                }
            }
            if (!options.kv_next_reference_dir.empty()) {
                const fs::path refdir(options.kv_next_reference_dir);
                if (!fs::is_directory(refdir))
                    throw std::runtime_error("Next Decode Result_0 not found: " +
                                             options.kv_next_reference_dir);
                for (size_t i = 0; i < decode->outputs.size(); ++i) {
                    const fs::path p = refdir / native_basename(decode->outputs[i],
                                                                static_cast<uint32_t>(i));
                    if (!fs::is_regular_file(p) ||
                        fs::file_size(p) != tensor_nbytes(decode->outputs[i]))
                        throw std::runtime_error("Invalid next Decode output reference: " +
                                                 p.string());
                    next_decode_references.push_back(p);
                }
            }
        }
        std::cout << "KV_MAPPING_OK: 30 cache/slice links; pos="
                  << (options.kv_auto_pos ? std::string("auto") :
                      std::to_string(options.kv_feedback_pos))
                  << "\n" << std::flush;
    }

    if (!options.cpu_aux_model.empty()) {
        if (jobs.front()->graph_name != "qnn_partition_0")
            throw std::runtime_error("--cpu-aux-model requires Decode-first jobs manifest");
        if (!decode) {
            for (auto& job : jobs) {
                if (job->graph_name != "qnn_partition_0") continue;
                if (decode) throw std::runtime_error("Duplicate Decode job for CPU AUX");
                decode = job.get();
            }
        }
        if (!decode) throw std::runtime_error("CPU_AUX_MISSING_DECODE_JOB");
        apply_cpu_aux_first_decode(*decode, options);
        if (options.kv_run_next)
            std::cout << "CPU_AUX_SCOPE_NOTE: first Decode AUX from CPU; "
                         "second Decode AUX from "
                      << (options.cpu_aux_next_pos_set ? "CPU Section1" : "saved next list")
                      << "; second-step embeddings from "
                      << (options.cpu_embed_next_token_set ? "CPU packed INT2/INT4" : "saved next list")
                      << "\n";
    }

    if (!options.cpu_embed_section0.empty()) {
        if (jobs.front()->graph_name != "qnn_partition_0")
            throw std::runtime_error(
                "--cpu-embed-section0 requires a Decode-first jobs manifest");
        // If AUX or KV linking has already selected the Decode job, reuse it.
        if (!decode) {
            for (auto& job : jobs) {
                if (job->graph_name != "qnn_partition_0") continue;
                if (decode) throw std::runtime_error(
                    "Duplicate Decode job for CPU embeddings");
                decode = job.get();
            }
        }
        if (!decode) throw std::runtime_error("CPU_EMBED_MISSING_DECODE_JOB");
        apply_cpu_embeddings_decode(*decode, options,
                                    options.cpu_embed_token, "first");
    }

    if (!options.cpu_prefill_tokens.empty()) {
        ReplayJob* prefill=nullptr;
        for(auto& job:jobs) {
            if(job->graph_name!="qnn_partition_2")continue;
            if(prefill)
                throw std::runtime_error("CPU_PREFILL_DUPLICATE_PREFILL_JOB");
            prefill=job.get();
        }
        if(!prefill)throw std::runtime_error("CPU_PREFILL_JOB_NOT_FOUND");
        // Keep Q8B's proven Decode-first loading order when multiple
        // contexts are present; Prefill-only replay is also supported.
        if(jobs.size()>1 && jobs.front()->graph_name!="qnn_partition_0")
            throw std::runtime_error("CPU_PREFILL_MULTI_CONTEXT_NEEDS_DECODE_FIRST");
        apply_cpu_prefill(*prefill,options);
    }

    QnnRuntime runtime(options.backend);
    // Critical property: load ALL contexts before executing ANY graph.
    for (auto& job : jobs) {
        const auto start = std::chrono::steady_clock::now();
        job->handle = runtime.load(job->binary, job->graph_name);
        const auto stop = std::chrono::steady_clock::now();
        std::cout << "QNN_CONTEXT_READY: " << job->graph_name
                  << " load_ms="
                  << std::chrono::duration<double, std::milli>(stop - start).count()
                  << "\n" << std::flush;
    }
    std::cout << "PERSISTENT_CONTEXTS_READY: " << jobs.size()
              << " graphs; one backend/device; all contexts alive\n" << std::flush;

    size_t total_outputs = 0;
    for (auto& job : jobs) {
        const auto start = std::chrono::steady_clock::now();
        runtime.execute(job->handle, job->inputs, job->outputs);
        const auto stop = std::chrono::steady_clock::now();
        for (size_t i = 0; i < job->references.size(); ++i) {
            // This checks all bytes and throws at the first mismatch.
            compare_reference(job->output_buffers[i], job->references[i].string());
            ++total_outputs;
        }
        std::cout << "PERSISTENT_GRAPH_OK: " << job->graph_name
                  << " outputs=" << job->outputs.size() << " execute_ms="
                  << std::chrono::duration<double, std::milli>(stop - start).count()
                  << "\n" << std::flush;
    }
    std::cout << "ALL_JOBS_REFERENCES_MATCH: " << jobs.size()
              << " graphs " << total_outputs << " output tensors\n";
    std::vector<std::vector<uint8_t>> linked_cache_buffers;
    if (options.prefill_kv_probe) {
        ReplayJob* src=nullptr;
        ReplayJob* dst=nullptr;
        for (auto& job:jobs) {
            if(job->graph_name=="qnn_partition_2") {
                if(src)throw std::runtime_error("KV_PREFILL_PROBE_DUPLICATE_PREFILL");
                src=job.get();
            } else if(job->graph_name=="qnn_partition_0") {
                if(dst)throw std::runtime_error("KV_PREFILL_PROBE_DUPLICATE_DECODE");
                dst=job.get();
            }
        }
        if(!src || !dst)
            throw std::runtime_error("KV_PREFILL_PROBE_NEEDS_PREFILL_AND_DECODE");
        probe_prefill_kv_bridge(*src,*dst,options.cpu_prefill_tokens,
                                options.prefill_kv_run ? &linked_cache_buffers : nullptr,
                                options.prefill_seed_last_prompt,
                                options.prefill_seed_first_prompt);
    }
    // Execute the linked graph *after* checking all saved individual baselines.
    // The saved Head reference belongs to a different Decode step; do not
    // incorrectly compare it to the newly chained Head output.
    if (options.chain_decode_head)
        execute_chained_decode_head(runtime, *decode, *head, hidden_output_index);
    if (options.kv_feedback) {
        const uint32_t position = options.kv_auto_pos ?
            infer_decode_kv_position(*decode, kv_links, next_cache_files) :
            options.kv_feedback_pos;
        apply_kv_feedback(*decode, kv_links, position, next_cache_files);
        if (options.kv_run_next) {
            execute_next_decode(runtime, *decode, kv_links, next_cache_files,
                                next_decode_references, options);
            if (options.chain_decode_head) {
                execute_chained_decode_head(runtime, *decode, *head, hidden_output_index);
                std::cout << "DECODE_NEXT_HEAD_EXECUTE_OK: logits="
                          << head->output_buffers[0].size()
                          << "\n" << std::flush;
                if (!next_head_reference.empty()) {
                    compare_reference(head->output_buffers[0],
                                      next_head_reference.string());
                    std::cout << "DECODE_NEXT_HEAD_REFERENCE_MATCH: "
                              << head->output_buffers[0].size()
                              << " logits bytes identical"
                              << "\n" << std::flush;
                }
            }
        }
    }
    if(options.prefill_kv_run) {
        ReplayJob* prefill=nullptr;
        for(auto& job:jobs)
            if(job->graph_name=="qnn_partition_2")prefill=job.get();
        if(!prefill || !decode || !head)
            throw std::runtime_error("PREFILL_LINK_MISSING_GRAPH");
        execute_prefill_linked_decode(
            runtime,*prefill,*decode,*head,hidden_output_index,options,
            std::move(linked_cache_buffers));
    }
    return 0;
}

int run(const Options& options) {
    if (!options.jobs.empty()) return run_replay_jobs(options);
    // Context metadata owns tensor names, dimensions and quantization arrays.
    // Keep the System context and source binary alive through graphExecute.
    auto binary = read_file(options.binary);
    std::cout << "BINARY " << options.binary << " bytes=" << binary.size() << "\n";
    SystemMetadata system(options.system);
    system.parse(binary);
    if (options.inspect) {
        system.inspect();
        return 0;
    }
    const GraphView& g = system.find(options.graph);
    std::cout << "GRAPH " << g.name << " inputs=" << g.num_inputs
              << " outputs=" << g.num_outputs << "\n";
    if (!options.input_list.empty() || !options.inputs_dir.empty()) {
        run_multi(options, g, binary);
        return 0;
    }
    if (g.num_inputs != 1 || g.num_outputs != 1)
        throw std::runtime_error("Stage 2 supports exactly 1 input and 1 output; "
                                 "use --inspect to see graph I/O");
    describe_tensor(g.inputs[0], "input", 0);
    describe_tensor(g.outputs[0], "output", 0);
    const size_t required = tensor_nbytes(g.inputs[0]);
    auto input = read_file(options.input);
    if (input.size() != required)
        throw std::runtime_error("INPUT_SIZE_MISMATCH: expected " +
                                 std::to_string(required) + " bytes, got " +
                                 std::to_string(input.size()));
    std::vector<uint8_t> output(tensor_nbytes(g.outputs[0]), 0);
    std::vector<Qnn_Tensor_t> input_tensors{g.inputs[0]};
    std::vector<Qnn_Tensor_t> output_tensors{g.outputs[0]};
    bind_raw(input_tensors[0], input);
    bind_raw(output_tensors[0], output);

    QnnRuntime runtime(options.backend);
    const auto load_start = std::chrono::steady_clock::now();
    Qnn_GraphHandle_t graph = runtime.load(binary, g.name);
    const auto load_end = std::chrono::steady_clock::now();
    const auto exec_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < options.repeat; ++i)
        runtime.execute(graph, input_tensors, output_tensors);
    const auto exec_end = std::chrono::steady_clock::now();

    // Save the result even when a later reference comparison fails.
    write_file(options.output, output);
    const double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count();
    const double exec_ms =
        std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
    std::cout << std::fixed << std::setprecision(3)
              << "QNN_EXECUTE_OK: " << g.name
              << " load_ms=" << load_ms
              << " avg_execute_ms=" << exec_ms / options.repeat
              << " repeat=" << options.repeat
              << " output=" << options.output
              << " bytes=" << output.size() << "\n";
    compare_reference(output, options.reference);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        usage(argv[0]);
        return 2;
    }
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "QNN_RUN_ERROR: " << e.what() << "\n";
        return 1;
    }
}
