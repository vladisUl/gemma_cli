// Gemma4 direct-QNN runtime probe (stage 1).
// Uses QAIRT SDK headers from the user's installed QAIRT 2.47; no Genie.
#include <QNN/QnnInterface.h>

#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(Qnn_ErrorHandle_t err, const std::string &what) {
    if (err != QNN_SUCCESS) {
        throw std::runtime_error(what + " failed, QNN error=" + std::to_string(err));
    }
}

std::vector<uint8_t> read_binary(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open context: " + path);
    const auto size = f.tellg();
    if (size <= 0) throw std::runtime_error("Empty context: " + path);
    if (static_cast<uint64_t>(size) > SIZE_MAX)
        throw std::runtime_error("Context too large for address space");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Cannot read context: " + path);
    return bytes;
}

struct Backend {
    void *lib = nullptr;
    const QnnInterface_t *provider = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    bool device_created = false;

    explicit Backend(const std::string &libpath) {
        lib = dlopen(libpath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!lib) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        auto getter = reinterpret_cast<decltype(&QnnInterface_getProviders)>(
            dlsym(lib, "QnnInterface_getProviders"));
        if (!getter) throw std::runtime_error("No QnnInterface_getProviders in backend");
        const QnnInterface_t **providers = nullptr;
        uint32_t count = 0;
        check(getter(&providers, &count), "QnnInterface_getProviders");
        if (!providers || !count) throw std::runtime_error("QNN backend has no providers");
        // For QAIRT SDK 2.47 the provider and compiled headers must be compatible.
        provider = providers[0];
        std::cout << "Provider: " << (provider->providerName ? provider->providerName : "(unnamed)")
                  << ", QNN API " << provider->apiVersion.coreApiVersion.major << "."
                  << provider->apiVersion.coreApiVersion.minor << "\n";
        const auto &api = provider->QNN_INTERFACE_VER_NAME;
        if (!api.backendCreate || !api.contextCreateFromBinary ||
            !api.graphRetrieve || !api.contextFree || !api.backendFree)
            throw std::runtime_error("QNN provider lacks required direct APIs");
        check(api.backendCreate(nullptr, nullptr, &backend), "backendCreate");
        if (api.deviceCreate) {
            const auto e = api.deviceCreate(nullptr, nullptr, &device);
            if (e == QNN_SUCCESS) {
                device_created = true;
            } else if (e != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
                check(e, "deviceCreate");
            }
        }
    }

    ~Backend() {
        if (provider && backend) {
            const auto &api = provider->QNN_INTERFACE_VER_NAME;
            if (device_created && device && api.deviceFree) api.deviceFree(device);
            if (api.backendFree) api.backendFree(backend);
        }
        if (lib) dlclose(lib);
    }

    Backend(const Backend &) = delete;
    Backend &operator=(const Backend &) = delete;

    void probe(const std::string &binary_path, const std::string &graph_name) {
        const auto &api = provider->QNN_INTERFACE_VER_NAME;
        std::cout << "Reading " << binary_path << " ..." << std::flush;
        auto bytes = read_binary(binary_path);
        std::cout << " " << bytes.size() / (1024.0 * 1024.0) << " MiB\n";
        Qnn_ContextHandle_t ctx = nullptr;
        Qnn_GraphHandle_t graph = nullptr;
        auto start = std::chrono::steady_clock::now();
        const auto e = api.contextCreateFromBinary(backend, device, nullptr,
            bytes.data(), static_cast<Qnn_ContextBinarySize_t>(bytes.size()),
            &ctx, nullptr);
        check(e, "contextCreateFromBinary(" + binary_path + ")");
        try {
            check(api.graphRetrieve(ctx, graph_name.c_str(), &graph),
                  "graphRetrieve(" + graph_name + ")");
        } catch (...) {
            api.contextFree(ctx, nullptr);
            throw;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "QNN_CONTEXT_OK: " << graph_name << " (" << ms << " ms)\n";
        check(api.contextFree(ctx, nullptr), "contextFree(" + graph_name + ")");
    }
};

} // namespace

int main(int argc, char **argv) {
    if (argc != 4 && argc != 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <libQnnHtp.so> [<context.bin> <graph-name>]\n"
                  << "Example: " << argv[0]
                  << " ./sdk247/libQnnHtp.so ./qnn_partition_1_1.bin qnn_partition_1\n";
        return 2;
    }
    try {
        Backend b(argv[1]);
        if (argc == 4) b.probe(argv[2], argv[3]);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "QNN_PROBE_ERROR: " << e.what() << "\n";
        return 1;
    }
}
