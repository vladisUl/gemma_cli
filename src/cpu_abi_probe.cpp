// Runtime C ABI smoke-test for the installed TFLite library.
// We intentionally do not include TensorFlow headers or load model files.
#include <dlfcn.h>

#include <array>
#include <iostream>
#include <string>

namespace {
constexpr const char* kDefaultLibrary =
    "/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1";

// C API functions needed to run the named signatures in Gemma 4 Section0/1/2.
constexpr std::array<const char*, 27> kRequired = {{
    "TfLiteVersion",
    "TfLiteModelCreateFromFile",
    "TfLiteModelDelete",
    "TfLiteInterpreterOptionsCreate",
    "TfLiteInterpreterOptionsDelete",
    "TfLiteInterpreterOptionsSetNumThreads",
    "TfLiteInterpreterCreate",
    "TfLiteInterpreterDelete",
    "TfLiteInterpreterGetSignatureCount",
    "TfLiteInterpreterGetSignatureKey",
    "TfLiteInterpreterGetSignatureRunner",
    "TfLiteSignatureRunnerDelete",
    "TfLiteSignatureRunnerGetInputCount",
    "TfLiteSignatureRunnerGetInputName",
    "TfLiteSignatureRunnerGetInputTensor",
    "TfLiteSignatureRunnerGetOutputCount",
    "TfLiteSignatureRunnerGetOutputName",
    "TfLiteSignatureRunnerGetOutputTensor",
    "TfLiteSignatureRunnerResizeInputTensor",
    "TfLiteSignatureRunnerAllocateTensors",
    "TfLiteSignatureRunnerInvoke",
    "TfLiteTensorCopyFromBuffer",
    "TfLiteTensorCopyToBuffer",
    "TfLiteTensorType",
    "TfLiteTensorByteSize",
    "TfLiteTensorNumDims",
    "TfLiteTensorDim",
}};

constexpr std::array<const char*, 2> kOptional = {{
    "TfLiteInterpreterCancel",
    "TfLiteSignatureRunnerCancel",
}};
}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: gemma4-cpu-abi [path/to/libtensorflow-lite.so]\n";
        return 2;
    }
    const std::string path = argc == 2 ? argv[1] : kDefaultLibrary;
    std::cout << "CPU_C_ABI_LOAD: " << path << '\n';
    void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::cerr << "CPU_C_ABI_DLOPEN_FAILED: " << dlerror() << '\n';
        return 1;
    }
    using VersionFn = const char* (*)();
    auto version = reinterpret_cast<VersionFn>(dlsym(lib, "TfLiteVersion"));
    if (version) {
        const char* v = version();
        std::cout << "CPU_C_ABI_VERSION: " << (v ? v : "(null)") << '\n';
    }

    std::size_t missing = 0;
    for (const auto* name : kRequired) {
        const bool exists = dlsym(lib, name) != nullptr;
        std::cout << (exists ? "CPU_C_ABI_FOUND: " : "CPU_C_ABI_MISSING: ")
                  << name << '\n';
        missing += !exists;
    }
    for (const auto* name : kOptional) {
        std::cout << (dlsym(lib, name) ? "CPU_C_ABI_OPTIONAL_FOUND: "
                                         : "CPU_C_ABI_OPTIONAL_MISSING: ")
                  << name << '\n';
    }
    dlclose(lib);
    if (missing) {
        std::cerr << "CPU_C_ABI_INCOMPLETE: missing=" << missing
                  << " required=" << kRequired.size() << '\n';
        return 1;
    }
    std::cout << "CPU_C_ABI_READY: required=" << kRequired.size()
              << " missing=0 signature_runner_available=1\n";
    std::cout << "CPU_C_ABI_NOTE: C symbols resolve; loading a real Section1 model is a separate test\n";
    return 0;
}
