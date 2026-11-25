// Isolated Stage 6.2 Prefill-128 embedding smoke test.
// Source model is mmap-read-only. Both output tensors are independently
// compared byte-for-byte against original QNN saved input references.
#include "packed_embedder.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs=std::filesystem;
void check(bool valid,const std::string& message) {
    if(!valid)throw std::runtime_error("PACKED_PREFILL_ERROR: "+message);
}
std::vector<uint8_t> read_file(const std::string& name,size_t expected) {
    check(fs::is_regular_file(name),"input file missing: "+name);
    check(fs::file_size(name)==expected,"unexpected input size: "+name);
    std::ifstream file(name,std::ios::binary);
    check(static_cast<bool>(file),"cannot read: "+name);
    std::vector<uint8_t> data(expected);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    check(static_cast<bool>(file),"short input read: "+name);
    return data;
}
struct Options {
    std::string model,tokens,reference,output;
    int section=-1;
};
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--help"||a=="-h") {
            std::cout<<"Usage: gemma4-packed-prefill --section 0|2 --model "
                        "MODEL.tflite --tokens TOKENS_I32_512.raw "
                        "--reference QNN_FLOAT32.raw --output OUTPUT.raw\n"
                        "Validate original 128-position INT2/INT4 embeddings "
                        "against a saved QNN input; no TF Lite runtime.\n";
            std::exit(0);
        }
        check(i+1<argc,"missing argument value: "+a);
        const std::string value=argv[++i];
        if(a=="--section") {
            check(o.section<0 && (value=="0"||value=="2"),
                  "--section must be exactly 0 or 2");
            o.section=value=="0"?0:2;
        } else if(a=="--model") o.model=value;
        else if(a=="--tokens")o.tokens=value;
        else if(a=="--reference")o.reference=value;
        else if(a=="--output")o.output=value;
        else check(false,"unknown option: "+a);
    }
    check((o.section==0||o.section==2) &&
          !o.model.empty()&&!o.tokens.empty()&&
          !o.reference.empty()&&!o.output.empty(),
          "--model, --tokens, --reference, --output and --section required");
    check(!fs::exists(o.output),"output exists; refusing overwrite");
    check(o.model!=o.output && o.tokens!=o.output && o.reference!=o.output,
          "output cannot overwrite a model or reference");
    return o;
}
} // namespace

int main(int argc,char** argv) {
    try {
        const Options o=parse(argc,argv);
        const auto token_bytes=read_file(o.tokens,128*sizeof(int32_t));
        std::vector<int32_t> tokens(128);
        // Q8B/aarch64 uses little-endian 32-bit int, like original TFLite.
        std::memcpy(tokens.data(),token_bytes.data(),token_bytes.size());
        const auto output=gemma4::compute_packed_prefill_embeddings(
            o.model,o.section,tokens);
        const size_t expected=o.section==0?786432:4587520;
        check(output.size()==expected,"wrong computed FLOAT32 output length");
        const auto reference=read_file(o.reference,expected);
        auto mismatch=std::mismatch(
            output.begin(),output.end(),reference.begin());
        if(mismatch.first!=output.end()) {
            const size_t byte=static_cast<size_t>(mismatch.first-output.begin());
            std::cout<<"PACKED_PREFILL_REFERENCE_MISMATCH: section="<<o.section
                     <<" first_byte="<<byte
                     <<" token_position="<<(byte/(expected/128))
                     <<" float_index="<<(byte/sizeof(float))<<"\n";
            return 1;
        }
        std::cout<<"PACKED_PREFILL_REFERENCE_MATCH: section="<<o.section
                 <<" bytes="<<expected<<" positions=128\n";
        std::ofstream save(o.output,std::ios::binary|std::ios::trunc);
        check(static_cast<bool>(save),"cannot create output: "+o.output);
        save.write(reinterpret_cast<const char*>(output.data()),
                   static_cast<std::streamsize>(output.size()));
        check(static_cast<bool>(save),"failed to write output");
        std::cout<<"PACKED_PREFILL_OK: section="<<o.section
                 <<" bytes="<<output.size()
                 <<" output="<<o.output<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<"\n";
        return 1;
    }
}
