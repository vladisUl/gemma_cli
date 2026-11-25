// Stage 2: execute a precompiled QNN context directly (without Genie).
// Uses QNN System metadata as the source of truth for graph I/O.
// Intended for QAIRT 2.47 on Radxa Q8B; first target is Gemma 4 LM Head.
#include <QNN/QnnInterface.h>
#include <QNN/System/QnnSystemInterface.h>

#include <dlfcn.h>

#include <algorithm>
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
    Qnn_ContextHandle_t context = nullptr;
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
        if (context && api.contextFree) api.contextFree(context, nullptr);
        if (device_created && device && api.deviceFree) api.deviceFree(device);
        if (backend && api.backendFree) api.backendFree(backend);
    }

    Qnn_GraphHandle_t load(const std::vector<uint8_t>& bytes, const std::string& name) {
        const auto& api = provider->QNN_INTERFACE_VER_NAME;
        if (bytes.empty()) throw std::runtime_error("Empty QNN context binary");
        check(api.contextCreateFromBinary(backend, device, nullptr, bytes.data(),
                                           static_cast<Qnn_ContextBinarySize_t>(bytes.size()),
                                           &context, nullptr),
              "contextCreateFromBinary");
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
    std::string input_list, inputs_dir, outputs_dir, reference_dir;
    bool inspect = false;
    bool check_inputs = false;
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



int run(const Options& options) {
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
