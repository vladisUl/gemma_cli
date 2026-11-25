// Experimental native packed INT2/INT4 Gemma 4 decode embedder.
// Read original TFLite FlatBuffer in place (mmap); no TensorFlow dependency.
// Only the verified Gemma Section0 / Section2 decode graph topology is accepted.
// Output equivalence must be independently proven against reference tensors.
#include "packed_embedder.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("PACKED_ERROR: " + message);
}
struct Blob { size_t p=0, n=0; };

struct Flat {
    const uint8_t* ptr=nullptr;
    size_t length=0;
    explicit Flat(const fs::path& filename) {
        int fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("PACKED_ERROR: cannot open "+filename.string());
        struct stat st{};
        if (fstat(fd,&st)!=0 || st.st_size<12) {
            close(fd);
            throw std::runtime_error("PACKED_ERROR: invalid file size");
        }
        length=static_cast<size_t>(st.st_size);
        void* data=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);
        close(fd);
        if(data==MAP_FAILED)
            throw std::runtime_error("PACKED_ERROR: mmap failed (large model)");
        ptr=static_cast<const uint8_t*>(data);
        if(std::memcmp(ptr+4,"TFL3",4)!=0) {
            munmap(const_cast<uint8_t*>(ptr),length);
            ptr=nullptr;
            throw std::runtime_error("PACKED_ERROR: not a TFL3 FlatBuffer");
        }
    }
    ~Flat(){ if(ptr)munmap(const_cast<uint8_t*>(ptr),length); }
    Flat(const Flat&)=delete;
    Flat& operator=(const Flat&)=delete;

    void check(size_t p,size_t n) const {
        require(p<=length && n<=length-p,"invalid FlatBuffer offset");
    }
    template<typename T> T value(size_t p) const {
        check(p,sizeof(T));
        T v{};
        std::memcpy(&v,ptr+p,sizeof(T));
        return v;
    }
    uint8_t u8(size_t p)const{return value<uint8_t>(p);}
    int8_t i8(size_t p)const{return value<int8_t>(p);}
    uint16_t u16(size_t p)const{return value<uint16_t>(p);}
    uint32_t u32(size_t p)const{return value<uint32_t>(p);}
    int32_t i32(size_t p)const{return value<int32_t>(p);}
    int64_t i64(size_t p)const{return value<int64_t>(p);}
    float f32(size_t p)const{return value<float>(p);}
    size_t indirect(size_t p) const {
        const uint32_t off=u32(p);
        check(p,off);
        return p+off;
    }
    size_t field(size_t table,unsigned index)const {
        const int64_t vtable=static_cast<int64_t>(table)-i32(table);
        require(vtable>=0 && static_cast<uint64_t>(vtable)<length,
                "vtable offset invalid");
        const size_t vt=static_cast<size_t>(vtable);
        const uint16_t size=u16(vt);
        const size_t offset=4+2*static_cast<size_t>(index);
        if(offset+2>size)return 0;
        const uint16_t val=u16(vt+offset);
        if(!val)return 0;
        check(table,val);
        return table+val;
    }
    uint32_t scalar_u32(size_t table,unsigned index,uint32_t def=0)const {
        const size_t p=field(table,index);
        return p?u32(p):def;
    }
    int32_t scalar_i32(size_t table,unsigned index,int32_t def=0)const {
        const size_t p=field(table,index);
        return p?i32(p):def;
    }
    int8_t scalar_i8(size_t table,unsigned index,int8_t def=0)const {
        const size_t p=field(table,index);
        return p?i8(p):def;
    }
    Blob vec(size_t table,unsigned index,size_t elementSize)const {
        const size_t p=field(table,index);
        if(!p)return {};
        const size_t start=indirect(p);
        const size_t n=u32(start);
        require(n<=length/elementSize,"invalid FlatBuffer vector length");
        check(start+4,n*elementSize);
        return {start+4,n};
    }
    std::vector<size_t> tables(size_t table,unsigned fieldIndex,size_t limit=100000)const{
        const Blob b=vec(table,fieldIndex,4);
        require(b.n<=limit,"too many tables");
        std::vector<size_t> result;
        result.reserve(b.n);
        for(size_t i=0;i<b.n;++i)
            result.push_back(indirect(b.p+i*4));
        return result;
    }
    std::vector<int32_t> ints(size_t table,unsigned fieldIndex,size_t limit=100000)const {
        const Blob b=vec(table,fieldIndex,4);
        require(b.n<=limit,"too many indices");
        std::vector<int32_t> result;
        result.reserve(b.n);
        for(size_t i=0;i<b.n;++i)
            result.push_back(i32(b.p+i*4));
        return result;
    }
    std::string string(size_t table,unsigned fieldIndex)const{
        const Blob b=vec(table,fieldIndex,1);
        require(b.n<=4096,"oversized FlatBuffer string");
        return std::string(reinterpret_cast<const char*>(ptr+b.p),b.n);
    }
};

struct Model {
    Flat f;
    size_t root=0,graph=0;
    std::vector<size_t> buffers,tensors,operators,codes;
    int section=0,token=0;

    Model(const fs::path& path,int sec,int tok)
        :f(path),section(sec),token(tok) {
        require(section==0||section==2,"section must be 0 or 2");
        require(token>=0&&token<262144,"token must be within 0..262143");
        root=f.u32(0);
        require(f.scalar_u32(root,0)==3,"unexpected TFLite schema version");
        codes=f.tables(root,1);
        auto graphs=f.tables(root,2);
        require(graphs.size()==2,"expected exactly two decode/prefill graphs");
        graph=graphs.at(0);
        buffers=f.tables(root,4,1000000);
        tensors=f.tables(graph,0,1000000);
        operators=f.tables(graph,3,10000);
        auto ins=f.ints(graph,1),outs=f.ints(graph,2);
        require(ins.size()==1 && outs.size()==1,"expected single input/output decode embedder");
        require(tensorType(ins[0])==2,"expected INT32 token input");
        require(tensorType(outs[0])==0,"expected FLOAT32 embedding output");
        const std::vector<int32_t> inshape=f.ints(tensors.at(ins[0]),0);
        const std::vector<int32_t> expectedOutput=
            section==0?std::vector<int32_t>{1,1,1536}:
                       std::vector<int32_t>{1,1,35,256};
        require(inshape==std::vector<int32_t>{1,1},"unexpected token input shape");
        require(f.ints(tensors.at(outs[0]),0)==expectedOutput,
                "unexpected output shape");
        require(operators.size()==static_cast<size_t>(section==0?8:42),
                "unexpected graph operator count");
    }
    int tensorType(int32_t idx)const {
        require(idx>=0&&static_cast<size_t>(idx)<tensors.size(),"invalid tensor index");
        return f.scalar_i8(tensors.at(idx),1);
    }
    Blob tensorData(int32_t idx)const{
        require(idx>=0&&static_cast<size_t>(idx)<tensors.size(),"invalid tensor index");
        const size_t tensor=tensors.at(idx);
        const size_t bufferId=f.scalar_u32(tensor,2);
        require(bufferId<buffers.size(),"invalid weight buffer index");
        return f.vec(buffers[bufferId],0,1);
    }
    int opCode(size_t op)const {
        const uint32_t index=f.scalar_u32(op,0);
        require(index<codes.size(),"invalid operator code index");
        const size_t code=codes.at(index);
        return std::max<int>(
            f.scalar_i8(code,0,0),f.scalar_i32(code,3,0));
    }
    int opVersion(size_t op)const{
        const uint32_t index=f.scalar_u32(op,0);
        return f.scalar_i32(codes.at(index),2,1);
    }
    std::vector<float> lookup(int32_t weightIndex)const{
        const size_t tensor=tensors.at(static_cast<size_t>(weightIndex));
        const int dtype=tensorType(weightIndex);
        const int expectedDtype=section==0?19:17; // INT2 / INT4
        const int columns=section==0?1536:256;
        require(dtype==expectedDtype,
                "unexpected embedding weight dtype (INT2=19 / INT4=17)");
        const std::vector<int32_t> shape=f.ints(tensor,0);
        require(shape==std::vector<int32_t>({262144,columns}),
                "wrong embedding weight matrix shape");
        const Blob packed=tensorData(weightIndex);
        const int bits=dtype==19?2:4;
        const size_t rowBytes=static_cast<size_t>(columns)*bits/8;
        require(packed.n==262144*rowBytes,"packed weight bytes mismatch");
        const size_t quantField=f.field(tensor,4);
        require(quantField!=0,"missing quantization");
        const size_t quant=f.indirect(quantField);
        const Blob scales=f.vec(quant,2,4);
        const Blob zero=f.vec(quant,3,8);
        require(scales.n==262144,"expected per-token 262144 scales");
        require(zero.n==1||zero.n==262144,"unexpected zero-point count");
        require(f.scalar_i32(quant,6,0)==0,"quantized dimension must be row 0");
        require(f.i64(zero.p+8*(zero.n==1?0:static_cast<size_t>(token)))==0,
                "only symmetric zero-point=0 supported");
        const float scale=f.f32(scales.p+4*static_cast<size_t>(token));
        const float factor=scale/static_cast<float>(1U<<(8-bits));
        const size_t rowStart=packed.p+static_cast<size_t>(token)*rowBytes;
        std::vector<float> output(columns);
        for(int j=0;j<columns;++j) {
            const uint8_t byte=f.u8(rowStart+static_cast<size_t>(j)*bits/8);
            const unsigned shift=static_cast<unsigned>(j*bits%8);
            const unsigned mask=(1U<<bits)-1;
            const int raw=static_cast<int>((byte>>shift)&mask);
            const int signedVal=raw>=static_cast<int>(1U<<(bits-1))?
                                raw-static_cast<int>(1U<<bits):raw;
            const int promoted=signedVal*(1U<<(8-bits));
            output[j]=static_cast<float>(promoted)*factor;
        }
        return output;
    }
    float constantFloat(int32_t index)const {
        require(tensorType(index)==0,"MUL constant must be FLOAT32");
        const Blob b=tensorData(index);
        require(b.n==4,"MUL expected exactly one FLOAT32 scalar");
        return f.f32(b.p);
    }
    std::vector<float> run()const {
        std::map<int32_t,std::vector<float>> produced;
        unsigned embedCount=0,mulCount=0,concatCount=0;
        for(const size_t op:operators) {
            const int code=opCode(op);
            const auto in=f.ints(op,1);
            const auto out=f.ints(op,2);
            if(code==7) {
                require(opVersion(op)==4 && in.size()==2 && out.size()==1,
                        "expected EMBEDDING_LOOKUP v4 with 2 inputs");
                produced[out[0]]=lookup(in[1]);
                ++embedCount;
            } else if(code==18 && section==0) {
                require(in.size()==2&&out.size()==1,"unexpected Section0 MUL");
                auto first=produced.find(in[0]);
                auto second=produced.find(in[1]);
                require((first!=produced.end()) != (second!=produced.end()),
                        "MUL must have one embedding input and one scalar constant");
                const auto& embedding=first!=produced.end()?first->second:second->second;
                const int scalarTensor=first!=produced.end()?in[1]:in[0];
                const float factor=constantFloat(scalarTensor);
                std::vector<float> v=embedding;
                for(float& x:v)x*=factor;
                produced[out[0]]=std::move(v);
                ++mulCount;
                std::cout<<"PACKED_EMBED_MUL: scalar="<<factor<<"\n";
            } else if(code==22) {
                // RESHAPE must forward the actual FLOAT32 embedding data.
                // The first RESHAPE in both models is on the INT32 token
                // selection path and can safely be ignored for a valid token.
                // Section0's final RESHAPE follows MUL and supplies the
                // signature output tensor; ignoring it dropped that output.
                require(in.size()>=1 && out.size()==1,
                        "unexpected RESHAPE signature");
                const auto value=produced.find(in[0]);
                if (value!=produced.end()) {
                    require(tensorType(in[0])==0 && tensorType(out[0])==0,
                            "embedding RESHAPE must preserve FLOAT32 dtype");
                    const auto targetShape=f.ints(tensors.at(
                        static_cast<size_t>(out[0])),0);
                    size_t elements=1;
                    for (int32_t dim:targetShape) {
                        require(dim>0 && elements<=std::numeric_limits<size_t>::max()/
                                                   static_cast<size_t>(dim),
                                "invalid target embedding RESHAPE dimension");
                        elements*=static_cast<size_t>(dim);
                    }
                    require(elements==value->second.size(),
                            "embedding RESHAPE changes element count");
                    produced[out[0]]=value->second;
                    std::cout<<"PACKED_EMBED_RESHAPE: elements="<<elements
                             <<" output_tensor="<<out[0]<<"\n";
                } else {
                    // Token-side RESHAPE; this helper already accepts only
                    // validated single-token IDs in [0, 262143].
                    require(tensorType(in[0])==2 && tensorType(out[0])==2,
                            "unexpected uncomputed non-token RESHAPE");
                }
            } else if(code==2 && section==2) {
                require(in.size()==35&&out.size()==1,"expected CONCATENATION of 35 vectors");
                const size_t optionsField=f.field(op,4);
                require(optionsField!=0,"CONCATENATION missing options");
                const size_t options=f.indirect(optionsField);
                int axis=f.scalar_i32(options,0);
                if(axis<0)axis+=4;
                // A single-token embedding is flattened in memory; all
                // concat ranks/axes allowed by the original 35-input graph
                // produce the same contiguous row-major 35x256 result.
                // We still require a valid four-or-fewer-dim axis.
                require(axis>=0&&axis<4,"invalid concat axis");
                std::cout<<"PACKED_EMBED_CONCAT: axis="<<axis
                         <<" inputs="<<in.size()<<"\n";
                std::vector<float> joined;
                joined.reserve(35*256);
                for(int32_t id:in) {
                    auto it=produced.find(id);
                    require(it!=produced.end()&&it->second.size()==256,
                            "unresolved concat embedding tensor");
                    joined.insert(joined.end(),it->second.begin(),it->second.end());
                }
                produced[out[0]]=std::move(joined);
                ++concatCount;
            } else {
                // Restrict the ignored subgraph to token-range guards and
                // identity reshaping. This path handles only valid token IDs.
                require(code==62||code==58||code==86||code==123,
                        "unknown or unsupported TFLite op in decode model");
                require(opVersion(op)==1,"unexpected guard/reshape operator version");
            }
        }
        require(embedCount==static_cast<unsigned>(section==0?1:35),
                "wrong number of embedded table lookups");
        require(mulCount==static_cast<unsigned>(section==0?1:0),
                "expected one Section0 multiply");
        require(concatCount==static_cast<unsigned>(section==2?1:0),
                "expected one Section2 concatenation");
        const auto outs=f.ints(graph,2);
        auto it=produced.find(outs.at(0));
        require(it!=produced.end(),"no computed output tensor for original graph");
        require(it->second.size()==static_cast<size_t>(section==0?1536:35*256),
                "wrong output size");
        return it->second;
    }
};

} // namespace

namespace gemma4 {
std::vector<uint8_t> compute_packed_embedding(
        const std::string& model_file, int section, int token) {
    const Model model(model_file, section, token);
    const std::vector<float> output = model.run();
    const size_t byte_size = output.size() * sizeof(float);
    std::vector<uint8_t> bytes(byte_size);
    std::memcpy(bytes.data(), output.data(), byte_size);
    std::cout << "PACKED_EMBED_RAM_READY: section=" << section
              << " token=" << token << " bytes=" << bytes.size()
              << "\n" << std::flush;
    return bytes;
}
// Prefill output tensors use token-major row order:
// Section0: [1,128,1536], Section2: [1,128,35,256].
// The original single-token extractor is reused for each distinct token,
// keeping one read-only mmap for the entire model and caching repeated IDs.
std::vector<uint8_t> compute_packed_prefill_embeddings(
        const std::string& model_file, int section,
        const std::vector<int32_t>& token_ids) {
    if (token_ids.size() != 128)
        throw std::runtime_error("PACKED_PREFILL_ERROR: expected 128 token IDs");
    for (auto token : token_ids)
        if (token < 0 || token >= 262144)
            throw std::runtime_error("PACKED_PREFILL_ERROR: invalid token ID");
    if (section != 0 && section != 2)
        throw std::runtime_error("PACKED_PREFILL_ERROR: section must be 0 or 2");
    Model model(model_file, section, token_ids.front());
    const size_t row_floats = section == 0 ? 1536 : 35*256;
    const size_t row_bytes = row_floats * sizeof(float);
    std::vector<uint8_t> result(token_ids.size() * row_bytes);
    std::map<int32_t, std::vector<float>> cache;
    for (size_t i=0;i<token_ids.size();++i) {
        const int32_t token = token_ids[i];
        auto entry=cache.find(token);
        if(entry==cache.end()) {
            model.token=token;
            auto fresh=model.run();
            if(fresh.size()!=row_floats)
                throw std::runtime_error("PACKED_PREFILL_ERROR: unexpected row length");
            entry=cache.emplace(token,std::move(fresh)).first;
        }
        std::memcpy(result.data()+i*row_bytes,
                    entry->second.data(),row_bytes);
    }
    std::cout<<"PACKED_PREFILL_RAM_READY: section="<<section
             <<" positions="<<token_ids.size()
             <<" unique_tokens="<<cache.size()
             <<" bytes="<<result.size()
             <<" mmap_count=1\n"<<std::flush;
    return result;
}
} // namespace gemma4

#ifndef GEMMA4_PACKED_EMBED_LIBRARY
namespace {
struct Args {
    std::string model,output,reference;
    int token=-1,section=-1;
};
Args parse(int argc,char** argv) {
    Args a;
    for(int i=1;i<argc;++i) {
        const std::string flag=argv[i];
        if(flag=="-h"||flag=="--help") {
            std::cout<<"Usage: gemma4-packed-embed --section 0|2 --model MODEL.tflite"
                        " --token N --output FILE.raw [--reference FILE.raw]\n"
                        "Experimental INT2/INT4 mmap lookup for Gemma 4 decode only.\n";
            std::exit(0);
        }
        require(i+1<argc,"missing value for "+flag);
        std::string value=argv[++i];
        if(flag=="--model")a.model=value;
        else if(flag=="--output")a.output=value;
        else if(flag=="--reference")a.reference=value;
        else if(flag=="--section"||flag=="--token") {
            size_t parsed=0;
            int number=std::stoi(value,&parsed);
            require(parsed==value.size(),"invalid number: "+value);
            if(flag=="--section")a.section=number;
            else a.token=number;
        } else require(false,"unknown argument "+flag);
    }
    require(!a.model.empty()&&!a.output.empty()&&a.token>=0&&
            (a.section==0||a.section==2),"missing required argument");
    require(!fs::exists(a.output),"output exists; choose a fresh path");
    require(!a.reference.empty()?a.reference!=a.output:true,
            "reference cannot be the output");
    return a;
}
Bytes readFile(const std::string& file) {
    std::ifstream s(file,std::ios::binary|std::ios::ate);
    require(static_cast<bool>(s),"cannot open reference");
    const std::streamoff size=s.tellg();
    require(size>=0 && size<=65536,"reference size invalid");
    Bytes result(static_cast<size_t>(size));
    s.seekg(0);
    if(!result.empty())
        s.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(size));
    require(static_cast<bool>(s)||result.empty(),"reference read failed");
    return result;
}
} // namespace

int main(int argc,char** argv) {
    try {
        const Args a=parse(argc,argv);
        const Model m(a.model,a.section,a.token);
        const auto floats=m.run();
        const size_t n=floats.size()*sizeof(float);
        Bytes data(n);
        std::memcpy(data.data(),floats.data(),n);
        std::cout<<"PACKED_EMBED_READY: section="<<a.section
                 <<" token="<<a.token
                 <<" floats="<<floats.size()
                 <<" bytes="<<n<<"\n";
        if(!a.reference.empty()) {
            const Bytes expected=readFile(a.reference);
            require(expected.size()==data.size(),"reference output size differs");
            if(data!=expected) {
                const auto mismatch=std::mismatch(data.begin(),data.end(),expected.begin());
                const size_t i=static_cast<size_t>(mismatch.first-data.begin());
                std::cout<<"PACKED_EMBED_REFERENCE_MISMATCH: first_byte="<<i
                         <<" actual="<<static_cast<unsigned>(data[i])
                         <<" expected="<<static_cast<unsigned>(expected[i])<<"\n";
                return 1;
            }
            std::cout<<"PACKED_EMBED_REFERENCE_MATCH: "<<n<<" bytes identical\n";
        } else {
            std::cout<<"PACKED_EMBED_NOTE: no reference supplied; byte accuracy unverified\n";
        }
        std::ofstream output(a.output,std::ios::binary|std::ios::trunc);
        require(static_cast<bool>(output),"cannot create output");
        output.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size()));
        require(static_cast<bool>(output),"writing output failed");
        std::cout<<"PACKED_EMBED_OK: "<<a.output<<" bytes="<<n<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<"\n";
        return 1;
    }
}

#endif // GEMMA4_PACKED_EMBED_LIBRARY
