#include "cpu_aux.h"

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
struct Library {
    void* handle = nullptr;
    explicit Library(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(
            std::string("CPU_AUX_DLOPEN_FAILED: ") + dlerror());
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    ~Library() { if (handle) dlclose(handle); }

    template<typename Function>
    Function symbol(const char* name) const {
        auto f = reinterpret_cast<Function>(dlsym(handle, name));
        if (!f) throw std::runtime_error(std::string("CPU_AUX_SYMBOL_MISSING: ")+name);
        return f;
    }
};

struct Api {
    const Library& library;
    const char* (*version)();
    void* (*create_model)(const char*);
    void (*delete_model)(void*);
    void* (*create_options)();
    void (*delete_options)(void*);
    void (*set_threads)(void*, int32_t);
    void* (*create_interpreter)(const void*, const void*);
    void (*delete_interpreter)(void*);
    void* (*get_signature)(const void*, const char*);
    void (*delete_signature)(void*);
    void* (*get_input)(void*, const char*);
    const void* (*get_output)(const void*, const char*);
    int (*allocate)(void*);
    int (*invoke)(void*);
    int (*copy_input)(void*, const void*, size_t);
    int (*copy_output)(const void*, void*, size_t);
    int (*tensor_type)(const void*);
    size_t (*tensor_bytes)(const void*);

    explicit Api(const Library& library)
        : library(library),
          version(library.symbol<decltype(version)>("TfLiteVersion")),
          create_model(library.symbol<decltype(create_model)>("TfLiteModelCreateFromFile")),
          delete_model(library.symbol<decltype(delete_model)>("TfLiteModelDelete")),
          create_options(library.symbol<decltype(create_options)>("TfLiteInterpreterOptionsCreate")),
          delete_options(library.symbol<decltype(delete_options)>("TfLiteInterpreterOptionsDelete")),
          set_threads(library.symbol<decltype(set_threads)>("TfLiteInterpreterOptionsSetNumThreads")),
          create_interpreter(library.symbol<decltype(create_interpreter)>("TfLiteInterpreterCreate")),
          delete_interpreter(library.symbol<decltype(delete_interpreter)>("TfLiteInterpreterDelete")),
          get_signature(library.symbol<decltype(get_signature)>("TfLiteInterpreterGetSignatureRunner")),
          delete_signature(library.symbol<decltype(delete_signature)>("TfLiteSignatureRunnerDelete")),
          get_input(library.symbol<decltype(get_input)>("TfLiteSignatureRunnerGetInputTensor")),
          get_output(library.symbol<decltype(get_output)>("TfLiteSignatureRunnerGetOutputTensor")),
          allocate(library.symbol<decltype(allocate)>("TfLiteSignatureRunnerAllocateTensors")),
          invoke(library.symbol<decltype(invoke)>("TfLiteSignatureRunnerInvoke")),
          copy_input(library.symbol<decltype(copy_input)>("TfLiteTensorCopyFromBuffer")),
          copy_output(library.symbol<decltype(copy_output)>("TfLiteTensorCopyToBuffer")),
          tensor_type(library.symbol<decltype(tensor_type)>("TfLiteTensorType")),
          tensor_bytes(library.symbol<decltype(tensor_bytes)>("TfLiteTensorByteSize")) {}
};

using Obj = std::unique_ptr<void, std::function<void(void*)>>;
Obj ownership(void* ptr, void (*destroy)(void*), const std::string& label) {
    if (!ptr) throw std::runtime_error("CPU_AUX_CREATE_FAILED: "+label);
    return Obj(ptr, [destroy](void* p) { if (p) destroy(p); });
}

struct Output {
    const char* cpu_name;
    const char* qnn_name;
    size_t expected_bytes;
};

using NamedInputs = std::vector<std::pair<std::string, int32_t>>;
using OutputList = std::vector<Output>;

void run_signature(
    const Api& api, void* interpreter, const char* signature,
    const NamedInputs& inputs, const OutputList& outputs,
    std::vector<gemma4::CpuAuxTensor>& result) {
    Obj runner = ownership(api.get_signature(interpreter, signature),
                           api.delete_signature, signature);
    if (api.allocate(runner.get()) != 0)
        throw std::runtime_error(std::string("CPU_AUX_ALLOCATE_FAILED: ")+signature);
    for (const auto& entry : inputs) {
        void* t = api.get_input(runner.get(), entry.first.c_str());
        if (!t || api.tensor_type(t) != 2 || api.tensor_bytes(t) != sizeof(int32_t))
            throw std::runtime_error("CPU_AUX_INT32_INPUT_MISMATCH: "+entry.first);
        if (api.copy_input(t, &entry.second, sizeof(int32_t)) != 0)
            throw std::runtime_error("CPU_AUX_INPUT_COPY_FAILED: "+entry.first);
    }
    if (api.invoke(runner.get()) != 0)
        throw std::runtime_error(std::string("CPU_AUX_INVOKE_FAILED: ")+signature);
    for (const auto& spec : outputs) {
        const void* t = api.get_output(runner.get(), spec.cpu_name);
        if (!t || api.tensor_type(t) != 9 ||
            api.tensor_bytes(t) != spec.expected_bytes)
            throw std::runtime_error(
                std::string("CPU_AUX_INT8_OUTPUT_MISMATCH: ")+spec.cpu_name);
        gemma4::CpuAuxTensor value;
        value.qnn_name = spec.qnn_name;
        value.bytes.resize(spec.expected_bytes);
        if (api.copy_output(t, value.bytes.data(), value.bytes.size()) != 0)
            throw std::runtime_error(
                std::string("CPU_AUX_OUTPUT_COPY_FAILED: ")+spec.cpu_name);
        std::cout << "CPU_AUX_OUTPUT_READY: "<< spec.cpu_name
                  << " qnn=" << value.qnn_name
                  << " bytes=" << value.bytes.size() << "\n" << std::flush;
        result.push_back(std::move(value));
    }
    std::cout << "CPU_AUX_SIGNATURE_OK: " << signature
              << " outputs=" << outputs.size() << "\n" << std::flush;
}
} // namespace

namespace gemma4 {
std::vector<CpuAuxTensor> run_decode_aux(
    const std::string& model_file, int32_t position, int32_t token_id,
    const std::string& cpu_library) {
    if (!std::filesystem::is_regular_file(model_file))
        throw std::runtime_error("CPU_AUX_MODEL_NOT_FOUND: "+model_file);
    if (position < 0 || position >= 4096 || token_id < 0 || token_id >= 262144)
        throw std::runtime_error("CPU_AUX_INPUT_RANGE_INVALID");
    Library library(cpu_library);
    Api api(library);
    std::cout << "CPU_AUX_VERSION: " << api.version()
              << " position=" << position
              << " token_id=" << token_id << "\n" << std::flush;

    Obj model = ownership(api.create_model(model_file.c_str()),
                          api.delete_model, "Section1 model");
    Obj options = ownership(api.create_options(), api.delete_options,
                            "interpreter options");
    api.set_threads(options.get(), 1);
    Obj interpreter = ownership(
        api.create_interpreter(model.get(),options.get()),
        api.delete_interpreter, "Section1 interpreter");

    std::vector<CpuAuxTensor> result;
    result.reserve(6);
    // The QNN exported Decode inputs use these exact names.
    // Output tensor lookup is by TFLite signature name, never graph index.
    run_signature(api, interpreter.get(), "decode_rope",
        {{"input_pos",position}},
        {{"pos_emb_cos","decode_pos_emb_cos_0",512},
         {"pos_emb_sin","decode_pos_emb_sin_0",512},
         {"pos_emb_local_sin","decode_pos_emb_local_sin_0",256},
         {"pos_emb_local_cos","decode_pos_emb_local_cos_0",256}},
        result);
    run_signature(api, interpreter.get(), "decode_mask",
        {{"input_tokens",token_id},{"time_step",position}},
        {{"mask_local","decode_mask_local_0",4097},
         {"mask_global","decode_mask_global_0",4097}},
        result);
    if (result.size()!=6) throw std::runtime_error("CPU_AUX_OUTPUT_COUNT_INVALID");
    return result;
}
}
