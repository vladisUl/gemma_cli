// Stage 8.2: experimental HTP RPCmem registration for Decode KV ONLY.
// This include is compiled after ReplayJob, KvLink, and QnnRuntime.
// Never applied to Prefill, LM Head, or ordinary text-generation paths.
// The CPU cache remains ground truth and we mirror ONLY the changed slot.
class Stage82KvMem {
    struct Buffer {
        size_t input_index=0;
        uint32_t channels=0;
        bool key=false;
        uint8_t* pointer=nullptr;
        size_t bytes=0;
        Qnn_MemHandle_t handle=nullptr;
    };
    QnnRuntime& runtime_;
    ReplayJob& decode_;
    std::unique_ptr<SharedLibrary> library_;
    using AllocFn=void*(*)(int,uint32_t,int);
    using FreeFn=void(*)(void*);
    using FdFn=int(*)(void*);
    AllocFn alloc_=nullptr;
    FreeFn free_=nullptr;
    FdFn to_fd_=nullptr;
    std::vector<Buffer> buffers_;
    size_t registered_=0;
public:
    Stage82KvMem(QnnRuntime& runtime,ReplayJob& decode,
                 const std::vector<KvLink>& links):
        runtime_(runtime),decode_(decode) {
        // libcdsprpc must be provided by the BSP / QAIRT loader on this board.
        // If unavailable, fail this experiment, NEVER silently use RAW.
        const char* override=std::getenv("GEMMA4_RPCMEM_LIBRARY");
        const std::string name=override&&*override?override:"libcdsprpc.so";
        library_=std::make_unique<SharedLibrary>(name);
        alloc_=library_->symbol<AllocFn>("rpcmem_alloc");
        free_=library_->symbol<FreeFn>("rpcmem_free");
        to_fd_=library_->symbol<FdFn>("rpcmem_to_fd");
        const auto& api=runtime_.provider->QNN_INTERFACE_VER_NAME;
        if(!api.memRegister || !api.memDeRegister)
            throw std::runtime_error("STAGE82_MEM_REGISTER_APIS_UNAVAILABLE");
        buffers_.reserve(links.size());
        try {
            const auto ctx=runtime_.context_for(decode.handle);
            for(const auto& link:links) {
                const auto& t=decode.inputs.at(link.input_index);
                const auto& raw=decode.input_buffers.at(link.input_index);
                const size_t bytes=raw.size();
                if(bytes!=size_t(link.channels)*4096 || bytes==0 ||
                   bytes>static_cast<size_t>(std::numeric_limits<int>::max()) ||
                   tensor_dtype(t)!=QNN_DATATYPE_SFIXED_POINT_8)
                    throw std::runtime_error("STAGE82_KV_TENSOR_INVALID");
                Buffer b;
                b.input_index=link.input_index;
                b.channels=link.channels;
                b.key=link.key;
                b.bytes=bytes;
                // Qualcomm documented system heap (25), default flags (1).
                b.pointer=static_cast<uint8_t*>(
                    alloc_(25,1,static_cast<int>(bytes)));
                if(!b.pointer)
                    throw std::runtime_error("STAGE82_RPCMEM_ALLOC_FAILED");
                buffers_.push_back(b); // destructor owns pointer on every path
                auto& saved=buffers_.back();
                std::memcpy(saved.pointer,raw.data(),bytes);
                const int fd=to_fd_(saved.pointer);
                if(fd<0)
                    throw std::runtime_error("STAGE82_RPCMEM_TO_FD_FAILED");
                Qnn_MemDescriptor_t descriptor=QNN_MEM_DESCRIPTOR_INIT;
                descriptor.memShape.numDim=tensor_rank(t);
                descriptor.memShape.dimSize=const_cast<uint32_t*>(tensor_dims(t));
                descriptor.dataType=tensor_dtype(t);
                descriptor.memType=QNN_MEM_TYPE_ION;
                descriptor.ionInfo.fd=fd;
                check(api.memRegister(ctx,&descriptor,1,&saved.handle),
                      "STAGE82_QNN_MEM_REGISTER");
                if(!saved.handle)
                    throw std::runtime_error("STAGE82_QNN_REGISTER_NULL_HANDLE");
                ++registered_;
            }
        } catch(...) {
            release();
            throw;
        }
        if(registered_!=30)
            throw std::runtime_error("STAGE82_REGISTERED_KV_NOT_30");
        size_t sum=0;
        for(const auto& b:buffers_)sum+=b.bytes;
        std::cout<<"STAGE82_MEM_REGISTERED: tensors="<<registered_
                 <<" bytes="<<sum<<" source=rpcmem fd_handles=1"
                 <<" backend_io=memhandle decode_only=1"
                 <<" cpu_mirror_only_new_slot=1\n"<<std::flush;
        verify_equal("initial");
    }
    ~Stage82KvMem(){release();}
    Stage82KvMem(const Stage82KvMem&)=delete;
    Stage82KvMem& operator=(const Stage82KvMem&)=delete;
    void release() noexcept {
        const auto& api=runtime_.provider->QNN_INTERFACE_VER_NAME;
        for(auto& b:buffers_) {
            if(b.handle && api.memDeRegister)
                (void)api.memDeRegister(&b.handle,1);
            b.handle=nullptr;
            if(b.pointer && free_)free_(b.pointer);
            b.pointer=nullptr;
        }
        buffers_.clear();
        registered_=0;
    }
    void bind_all() {
        if(registered_!=30)
            throw std::runtime_error("STAGE82_KV_NOT_REGISTERED");
        for(const auto& b:buffers_) {
            auto& t=decode_.inputs.at(b.input_index);
            if(t.version==QNN_TENSOR_VERSION_1) {
                t.v1.memType=QNN_TENSORMEMTYPE_MEMHANDLE;
                t.v1.memHandle=b.handle;
            } else if(t.version==QNN_TENSOR_VERSION_2) {
                t.v2.memType=QNN_TENSORMEMTYPE_MEMHANDLE;
                t.v2.memHandle=b.handle;
            } else {
                throw std::runtime_error("STAGE82_UNKNOWN_TENSOR_VERSION");
            }
        }
    }
    void mirror_slot(const KvLink& link,const std::vector<uint8_t>& slice,
                     size_t position) {
        if(position>=4096 || slice.size()!=link.channels)
            throw std::runtime_error("STAGE82_MIRROR_SLOT_INVALID");
        for(auto& b:buffers_) {
            if(b.input_index!=link.input_index)continue;
            if(b.channels!=link.channels || b.key!=link.key)
                throw std::runtime_error("STAGE82_MIRROR_LAYOUT_CHANGED");
            if(link.key) {
                for(size_t c=0;c<link.channels;++c)
                    b.pointer[c*4096+position]=slice[c];
            } else {
                std::memcpy(b.pointer+position*link.channels,
                            slice.data(),slice.size());
            }
            return;
        }
        throw std::runtime_error("STAGE82_MIRROR_INDEX_MISSING");
    }
    void verify_equal(const char* stage) const {
        if(registered_!=30)
            throw std::runtime_error("STAGE82_VERIFY_BAD_HANDLE_COUNT");
        for(const auto& b:buffers_) {
            const auto& original=decode_.input_buffers.at(b.input_index);
            if(original.size()!=b.bytes ||
               std::memcmp(original.data(),b.pointer,b.bytes)!=0)
                throw std::runtime_error(
                    std::string("STAGE82_CPU_RPCMEM_KV_MISMATCH: ")+stage);
        }
        std::cout<<"STAGE82_KV_MIRROR_OK: stage="<<stage
                 <<" full_cache_equal=1 tensors=30\n"<<std::flush;
    }
};
