// Native CPU TensorFlow Lite C signature runner via dlopen, no TFLite headers.
// Target: Linux aarch64 libtensorflow-lite.so.2.14.1 on Radxa Q8B.
// C ABI types and call signatures checked against TensorFlow Lite 2.14.
// Supports list-only mode and named, raw or scalar int32 signature inputs.
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using TfLiteStatus = int;
using TfLiteType = int;
using Model = void;
using Interpreter = void;
using Options = void;
using Runner = void;
using Tensor = void;

struct Api {
    void* so = nullptr;
    const char* (*version)();
    Model* (*model_from_file)(const char*);
    void (*model_delete)(Model*);
    Options* (*options_create)();
    void (*options_delete)(Options*);
    void (*options_threads)(Options*, int32_t);
    Interpreter* (*interpreter_create)(const Model*, const Options*);
    void (*interpreter_delete)(Interpreter*);
    int32_t (*signature_count)(const Interpreter*);
    const char* (*signature_key)(const Interpreter*, int32_t);
    Runner* (*runner_get)(const Interpreter*, const char*);
    void (*runner_delete)(Runner*);
    size_t (*input_count)(const Runner*);
    const char* (*input_name)(const Runner*, int32_t);
    Tensor* (*input_tensor)(Runner*, const char*);
    size_t (*output_count)(const Runner*);
    const char* (*output_name)(const Runner*, int32_t);
    const Tensor* (*output_tensor)(const Runner*, const char*);
    TfLiteStatus (*allocate)(Runner*);
    TfLiteStatus (*invoke)(Runner*);
    TfLiteStatus (*copy_input)(Tensor*, const void*, size_t);
    TfLiteStatus (*copy_output)(const Tensor*, void*, size_t);
    TfLiteType (*tensor_type)(const Tensor*);
    size_t (*tensor_bytes)(const Tensor*);
    int32_t (*tensor_rank)(const Tensor*);
    int32_t (*tensor_dim)(const Tensor*, int32_t);

    template<typename Fn>
    Fn load(const char* name) {
        auto fn = reinterpret_cast<Fn>(dlsym(so, name));
        if (!fn) throw std::runtime_error(std::string("CPU_API_MISSING: ") + name);
        return fn;
    }

    explicit Api(const std::string& lib) {
        so = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!so) throw std::runtime_error(std::string("CPU_DLOPEN_ERROR: ") + dlerror());
        try {
            version = load<decltype(version)>("TfLiteVersion");
            model_from_file = load<decltype(model_from_file)>("TfLiteModelCreateFromFile");
            model_delete = load<decltype(model_delete)>("TfLiteModelDelete");
            options_create = load<decltype(options_create)>("TfLiteInterpreterOptionsCreate");
            options_delete = load<decltype(options_delete)>("TfLiteInterpreterOptionsDelete");
            options_threads = load<decltype(options_threads)>("TfLiteInterpreterOptionsSetNumThreads");
            interpreter_create = load<decltype(interpreter_create)>("TfLiteInterpreterCreate");
            interpreter_delete = load<decltype(interpreter_delete)>("TfLiteInterpreterDelete");
            signature_count = load<decltype(signature_count)>("TfLiteInterpreterGetSignatureCount");
            signature_key = load<decltype(signature_key)>("TfLiteInterpreterGetSignatureKey");
            runner_get = load<decltype(runner_get)>("TfLiteInterpreterGetSignatureRunner");
            runner_delete = load<decltype(runner_delete)>("TfLiteSignatureRunnerDelete");
            input_count = load<decltype(input_count)>("TfLiteSignatureRunnerGetInputCount");
            input_name = load<decltype(input_name)>("TfLiteSignatureRunnerGetInputName");
            input_tensor = load<decltype(input_tensor)>("TfLiteSignatureRunnerGetInputTensor");
            output_count = load<decltype(output_count)>("TfLiteSignatureRunnerGetOutputCount");
            output_name = load<decltype(output_name)>("TfLiteSignatureRunnerGetOutputName");
            output_tensor = load<decltype(output_tensor)>("TfLiteSignatureRunnerGetOutputTensor");
            allocate = load<decltype(allocate)>("TfLiteSignatureRunnerAllocateTensors");
            invoke = load<decltype(invoke)>("TfLiteSignatureRunnerInvoke");
            copy_input = load<decltype(copy_input)>("TfLiteTensorCopyFromBuffer");
            copy_output = load<decltype(copy_output)>("TfLiteTensorCopyToBuffer");
            tensor_type = load<decltype(tensor_type)>("TfLiteTensorType");
            tensor_bytes = load<decltype(tensor_bytes)>("TfLiteTensorByteSize");
            tensor_rank = load<decltype(tensor_rank)>("TfLiteTensorNumDims");
            tensor_dim = load<decltype(tensor_dim)>("TfLiteTensorDim");
        } catch (...) {
            dlclose(so);
            throw;
        }
    }
    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;
    ~Api() { if (so) dlclose(so); }
};

struct Args {
    std::string lib = "/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1";
    std::string model, signature, output_dir;
    bool list = false;
    int threads = 1;
    std::map<std::string,std::string> raw, i32, expect;
};

std::pair<std::string,std::string> assignment(const std::string& s) {
    auto at=s.find('=');
    if (at==std::string::npos || at==0 || at+1==s.size())
        throw std::runtime_error("Expected NAME=VALUE, got: "+s);
    return {s.substr(0,at),s.substr(at+1)};
}

Args parse(int argc,char** argv) {
    Args a;
    for(int i=1;i<argc;++i) {
        std::string flag=argv[i];
        if(flag=="--help"||flag=="-h") {
            std::cout <<
                "CPU TFLite signature runner (C API, no headers)\n"
                "  gemma4-cpu-run --model Section1*.tflite --list\n"
                "  gemma4-cpu-run --model Section1*.tflite --signature decode_rope"
                " --i32 input_pos=3 [--output-dir DIR]\n"
                "  gemma4-cpu-run --model Section1*.tflite --signature decode_mask"
                " --i32 input_tokens=2 --i32 time_step=3 [--output-dir DIR]\n"
                "Options: --lib FILE --threads N --input NAME=FILE.raw"
                " --i32 NAME=INTEGER --expect OUTPUT_NAME=FILE.raw --output-dir DIR\n"
                "Tensor shape and byte sizes are validated before execution;"
                " no implicit conversions.\n";
            std::exit(0);
        }
        if(flag=="--list"){a.list=true;continue;}
        if(i+1>=argc)throw std::runtime_error("Missing argument to "+flag);
        std::string val=argv[++i];
        if(flag=="--model")a.model=val;
        else if(flag=="--lib")a.lib=val;
        else if(flag=="--signature")a.signature=val;
        else if(flag=="--output-dir")a.output_dir=val;
        else if(flag=="--input"){auto p=assignment(val);if(!a.raw.emplace(p).second)throw std::runtime_error("Duplicate input "+p.first);}
        else if(flag=="--i32"){auto p=assignment(val);if(!a.i32.emplace(p).second)throw std::runtime_error("Duplicate i32 "+p.first);}
        else if(flag=="--expect"){auto p=assignment(val);if(!a.expect.emplace(p).second)throw std::runtime_error("Duplicate expectation "+p.first);}
        else if(flag=="--threads"){
            size_t consumed=0;
            int n=std::stoi(val,&consumed);
            if(consumed!=val.size()||n<1||n>32)throw std::runtime_error("--threads must be 1..32");
            a.threads=n;
        } else throw std::runtime_error("Unknown argument: "+flag);
    }
    if(a.model.empty())throw std::runtime_error("--model is required");
    if(a.list==!a.signature.empty())throw std::runtime_error("Use exactly one of --list or --signature NAME");
    if(a.list&&(!a.raw.empty()||!a.i32.empty()||!a.expect.empty()||!a.output_dir.empty()))
        throw std::runtime_error("--list cannot be combined with run inputs/outputs");
    for(const auto& p:a.i32)
        if(a.raw.count(p.first))throw std::runtime_error("Both --input and --i32 for "+p.first);
    return a;
}

std::vector<uint8_t> read(const fs::path& p, size_t size) {
    if (!fs::is_regular_file(p) || fs::file_size(p)!=size)
        throw std::runtime_error("CPU_INPUT_SIZE_MISMATCH: "+p.string()+" expected="+std::to_string(size));
    std::ifstream f(p,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot read "+p.string());
    std::vector<uint8_t> data(size);
    if(size && !f.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(size)))
        throw std::runtime_error("Short read: "+p.string());
    return data;
}
std::string safe(const std::string& name) {
    std::string s;
    for(char ch:name)
        s+=((ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z')||
            (ch>='0'&&ch<='9')||ch=='_'||ch=='-')?ch:'_';
    if(s.size()>140)s.resize(140);
    if(s.empty())throw std::runtime_error("Empty output tensor name");
    return s;
}
std::string shape(const Api& api,const Tensor* t) {
    auto rank=api.tensor_rank(t);
    if(rank<0||rank>10)throw std::runtime_error("Invalid TFLite tensor rank");
    std::string s="[";
    for(int32_t j=0;j<rank;++j) {
        int32_t d=api.tensor_dim(t,j);
        if(d<0)throw std::runtime_error("Unresolved TFLite dynamic tensor dimension");
        if(j)s+=",";
        s+=std::to_string(d);
    }
    return s+"]";
}

int main(int argc,char** argv) {
    try {
        Args a=parse(argc,argv);
        if(!fs::is_regular_file(a.model))
            throw std::runtime_error("CPU_MODEL_NOT_FOUND: "+a.model);
        Api api(a.lib);
        std::cout<<"CPU_TFLITE_VERSION: "<<api.version()<<'\n';
        using Ptr=std::unique_ptr<void,std::function<void(void*)>>;
        Ptr model(api.model_from_file(a.model.c_str()),[&](void* p){if(p)api.model_delete(p);});
        if(!model)throw std::runtime_error("CPU_MODEL_CREATE_FAILED: "+a.model);
        Ptr options(api.options_create(),[&](void* p){if(p)api.options_delete(p);});
        if(!options)throw std::runtime_error("CPU_OPTIONS_CREATE_FAILED");
        api.options_threads(options.get(),a.threads);
        Ptr interpreter(api.interpreter_create(model.get(),options.get()),
                        [&](void* p){if(p)api.interpreter_delete(p);});
        if(!interpreter)throw std::runtime_error("CPU_INTERPRETER_CREATE_FAILED");
        const int32_t count=api.signature_count(interpreter.get());
        if(count<0||count>100)throw std::runtime_error("CPU_INVALID_SIGNATURE_COUNT");
        std::cout<<"CPU_SIGNATURE_COUNT: "<<count<<'\n';
        std::vector<std::string> signatures;
        for(int32_t i=0;i<count;++i) {
            const char* key=api.signature_key(interpreter.get(),i);
            if(!key)throw std::runtime_error("CPU_INVALID_SIGNATURE_KEY");
            signatures.emplace_back(key);
            std::cout<<"CPU_SIGNATURE: "<<key<<'\n';
        }
        if(a.list) {
            std::cout<<"CPU_SIGNATURES_LISTED: "<<count<<'\n';
            return 0;
        }
        if(std::find(signatures.begin(),signatures.end(),a.signature)==signatures.end())
            throw std::runtime_error("CPU_SIGNATURE_NOT_FOUND: "+a.signature);
        Ptr runner(api.runner_get(interpreter.get(),a.signature.c_str()),
                   [&](void* p){if(p)api.runner_delete(p);});
        if(!runner)throw std::runtime_error("CPU_SIGNATURE_RUNNER_CREATE_FAILED");
        if(api.allocate(runner.get())!=0)
            throw std::runtime_error("CPU_SIGNATURE_ALLOCATE_FAILED");
        std::map<std::string,bool> seenInputs;
        size_t inCount=api.input_count(runner.get());
        if(inCount>100)throw std::runtime_error("Unexpected input count");
        for(size_t i=0;i<inCount;++i) {
            const char* n=api.input_name(runner.get(),static_cast<int32_t>(i));
            if(!n)throw std::runtime_error("Null input name");
            std::string name=n;
            Tensor* t=api.input_tensor(runner.get(),n);
            if(!t)throw std::runtime_error("CPU_INPUT_TENSOR_MISSING: "+name);
            const auto bytes=api.tensor_bytes(t);
            const auto dtype=api.tensor_type(t);
            std::cout<<"CPU_INPUT_META: "<<name<<" type="<<dtype
                     <<" shape="<<shape(api,t)<<" bytes="<<bytes<<'\n';
            auto src=a.raw.find(name), immediate=a.i32.find(name);
            if(src==a.raw.end()&&immediate==a.i32.end())
                throw std::runtime_error("CPU_REQUIRED_INPUT_MISSING: "+name);
            if(src!=a.raw.end()) {
                auto data=read(src->second,bytes);
                if(api.copy_input(t,data.data(),data.size())!=0)
                    throw std::runtime_error("CPU_INPUT_COPY_FAILED: "+name);
            } else {
                if(dtype!=2||bytes!=sizeof(int32_t))
                    throw std::runtime_error("CPU_I32_TYPE_OR_SIZE_MISMATCH: "+name);
                size_t consumed=0;
                long long value=std::stoll(immediate->second,&consumed);
                if(consumed!=immediate->second.size()||
                   value<std::numeric_limits<int32_t>::min()||
                   value>std::numeric_limits<int32_t>::max())
                    throw std::runtime_error("CPU_I32_INVALID: "+name);
                const int32_t arg=static_cast<int32_t>(value);
                if(api.copy_input(t,&arg,sizeof(arg))!=0)
                    throw std::runtime_error("CPU_I32_COPY_FAILED: "+name);
            }
            seenInputs[name]=true;
        }
        for(const auto& p:a.raw)
            if(!seenInputs.count(p.first))throw std::runtime_error("CPU_EXTRA_INPUT: "+p.first);
        for(const auto& p:a.i32)
            if(!seenInputs.count(p.first))throw std::runtime_error("CPU_EXTRA_I32: "+p.first);

        std::cout<<"CPU_INVOKE_BEGIN: "<<a.signature<<'\n'<<std::flush;
        if(api.invoke(runner.get())!=0)
            throw std::runtime_error("CPU_INVOKE_FAILED: "+a.signature);
        const auto outCount=api.output_count(runner.get());
        if(outCount>100)throw std::runtime_error("Unexpected output count");
        std::map<std::string,bool> seenOutputs;
        std::map<std::string,bool> filesWritten;
        for(size_t i=0;i<outCount;++i) {
            const char* n=api.output_name(runner.get(),static_cast<int32_t>(i));
            if(!n)throw std::runtime_error("Null output name");
            std::string name=n;
            const Tensor* t=api.output_tensor(runner.get(),n);
            if(!t)throw std::runtime_error("CPU_OUTPUT_TENSOR_MISSING: "+name);
            size_t bytes=api.tensor_bytes(t);
            std::vector<uint8_t> result(bytes);
            if(api.copy_output(t,result.data(),bytes)!=0)
                throw std::runtime_error("CPU_OUTPUT_COPY_FAILED: "+name);
            std::cout<<"CPU_OUTPUT: "<<name<<" type="<<api.tensor_type(t)
                     <<" shape="<<shape(api,t)<<" bytes="<<bytes<<'\n';
            auto target=a.expect.find(name);
            if(target!=a.expect.end()) {
                auto expected=read(target->second,bytes);
                auto mismatch=std::mismatch(result.begin(),result.end(),expected.begin());
                if(mismatch.first!=result.end())
                    throw std::runtime_error("CPU_REFERENCE_MISMATCH: "+name+
                        " first_byte="+std::to_string(mismatch.first-result.begin()));
                std::cout<<"CPU_REFERENCE_MATCH: "<<name<<" bytes="<<bytes<<'\n';
            }
            if(!a.output_dir.empty()) {
                const auto base=safe(name);
                if(!filesWritten.emplace(base,true).second)
                    throw std::runtime_error("CPU_DUPLICATE_OUTPUT_FILENAME: "+base);
                fs::create_directories(a.output_dir);
                fs::path dst=fs::path(a.output_dir)/(base+".raw");
                if(fs::exists(dst))
                    throw std::runtime_error("CPU_OUTPUT_EXISTS: "+dst.string()+
                        " (choose a new --output-dir)");
                std::ofstream f(dst,std::ios::binary);
                if(!f || (bytes&&!f.write(reinterpret_cast<const char*>(result.data()),
                                             static_cast<std::streamsize>(bytes))))
                    throw std::runtime_error("CPU_WRITE_FAILED: "+dst.string());
                std::cout<<"CPU_OUTPUT_FILE: "<<dst.string()<<'\n';
            }
            seenOutputs[name]=true;
        }
        for(const auto& p:a.expect)
            if(!seenOutputs.count(p.first))
                throw std::runtime_error("CPU_EXTRA_REFERENCE: "+p.first);
        std::cout<<"CPU_SIGNATURE_OK: "<<a.signature<<" inputs="<<inCount
                 <<" outputs="<<outCount<<'\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"CPU_RUN_ERROR: "<<e.what()<<'\n';
        return 1;
    }
}
