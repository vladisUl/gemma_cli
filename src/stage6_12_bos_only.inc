// Stage 6.12: independent Prefill(BOS-only) execution and causal row0 checks.
// Included by qnn_runner.cpp after ReplayJob, KvLink and QnnRuntime declarations.
// A = Prefill(BOS, 127 pad); B = verified Prefill(17 tokens);
// C = Stage 6.11 zero-history Decode(pos0,BOS).
// Never mutate baseline descriptors, output buffers, or on-disk reference RAW.
void run_stage6_12_bos_only(
    QnnRuntime& runtime, const ReplayJob& prefill, const ReplayJob& decode,
    const std::vector<KvLink>& links, const Options& options) {
    if(prefill.graph_name!="qnn_partition_2" ||
       prefill.inputs.size()!=36 || prefill.outputs.size()!=30 ||
       decode.graph_name!="qnn_partition_0" || links.size()!=30)
        throw std::runtime_error("STAGE6_12_GRAPH_METADATA_INVALID");
    const auto raw=read_file(options.cpu_prefill_tokens);
    if(raw.size()!=512)throw std::runtime_error("STAGE6_12_TOKENS_512_BYTES_REQUIRED");
    std::array<int32_t,128> original{};
    std::memcpy(original.data(),raw.data(),raw.size());
    if(original[0]!=2)throw std::runtime_error("STAGE6_12_BOS_ID2_REQUIRED");
    size_t valid=0;
    while(valid<128 && original[valid]!=0)++valid;
    if(valid!=17)throw std::runtime_error("STAGE6_12_ORIGINAL_17_TOKENS_REQUIRED");
    for(size_t i=valid;i<128;++i)
        if(original[i]!=0)throw std::runtime_error("STAGE6_12_PADDING_INVALID");

    std::vector<int32_t> bos(128,0);
    bos[0]=2;
    auto inputs=prefill.inputs; // 28 unmodified zero-KV pointers remain valid
    std::vector<std::vector<uint8_t>> cpu(inputs.size());
    std::set<size_t> assigned;
    const auto set=[&](size_t index,const char* expected,Qnn_DataType_t dtype,
                       size_t bytes,std::vector<uint8_t> generated) {
        if(index>=inputs.size() || !assigned.insert(index).second ||
           !tensor_name(inputs[index]) ||
           std::string(tensor_name(inputs[index]))!=expected ||
           tensor_dtype(inputs[index])!=dtype ||
           tensor_nbytes(inputs[index])!=bytes || generated.size()!=bytes)
            throw std::runtime_error(std::string("STAGE6_12_BAD_INPUT: ")+expected);
        cpu[index]=std::move(generated);
        bind_raw(inputs[index],cpu[index]);
    };
    set(32,"prefill_128_embeddings_0",QNN_DATATYPE_FLOAT_32,786432,
        gemma4::compute_packed_prefill_embeddings(
            options.cpu_prefill_section0,0,bos));
    set(7,"prefill_128_per_layer_embeddings_0",QNN_DATATYPE_FLOAT_32,4587520,
        gemma4::compute_packed_prefill_embeddings(
            options.cpu_prefill_section2,2,bos));
    auto aux=gemma4::run_prefill_aux(options.cpu_prefill_aux_model,bos);
    if(aux.size()!=6)throw std::runtime_error("STAGE6_12_AUX_COUNT_INVALID");
    struct Target {const char* name;size_t index,bytes;};
    const std::array<Target,6> spec{{
        {"prefill_128_pos_emb_cos_0",3,65536},
        {"prefill_128_pos_emb_sin_0",5,65536},
        {"prefill_128_pos_emb_local_sin_0",6,32768},
        {"prefill_128_pos_emb_local_cos_0",18,32768},
        {"prefill_128_mask_local_0",21,540672},
        {"prefill_128_mask_global_0",26,540672}
    }};
    for(size_t i=0;i<6;++i) {
        if(aux[i].qnn_name!=spec[i].name)
            throw std::runtime_error("STAGE6_12_AUX_NAME_INVALID");
        set(spec[i].index,spec[i].name,QNN_DATATYPE_SFIXED_POINT_8,
            spec[i].bytes,std::move(aux[i].bytes));
    }
    if(assigned.size()!=8)throw std::runtime_error("STAGE6_12_EXPECTED_8_NONKV");
    size_t kv_inputs=0;
    for(size_t i=0;i<inputs.size();++i) {
        if(assigned.count(i))continue;
        const char* name=tensor_name(inputs[i]);
        const auto& q=tensor_quant(inputs[i]);
        if(!name || std::string(name).find("prefill_128_kv_cache_")!=0 ||
           tensor_dtype(inputs[i])!=QNN_DATATYPE_SFIXED_POINT_8 ||
           tensor_nbytes(inputs[i])!=prefill.input_buffers[i].size() ||
           q.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           q.scaleOffsetEncoding.offset!=0 ||
           std::any_of(prefill.input_buffers[i].begin(),
                       prefill.input_buffers[i].end(),
                       [](uint8_t byte){return byte!=0;}))
            throw std::runtime_error("STAGE6_12_KV_MUST_BE_RAW_ZERO");
        ++kv_inputs;
    }
    if(kv_inputs!=28)throw std::runtime_error("STAGE6_12_EXPECTED_28_KV");
    std::cout<<"PREFILL_BOS_ONLY_INPUTS_READY: valid_tokens=1 original_tokens=17"
             <<" padded_tokens=127 nonkv=8 zero_kv=28 read_only=1\n"<<std::flush;

    auto outputs=prefill.outputs;
    std::vector<std::vector<uint8_t>> fresh(outputs.size());
    for(size_t i=0;i<outputs.size();++i) {
        fresh[i].resize(tensor_nbytes(outputs[i]));
        bind_raw(outputs[i],fresh[i]); // independent output buffers
    }
    const auto saved_prefill=prefill.output_buffers;
    const auto saved_decode=decode.output_buffers;
    const auto begin=std::chrono::steady_clock::now();
    runtime.execute(prefill.handle,inputs,outputs);
    const double ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    if(prefill.output_buffers!=saved_prefill ||
       decode.output_buffers!=saved_decode)
        throw std::runtime_error("STAGE6_12_BASELINE_OUTPUT_MUTATED");
    std::cout<<"PREFILL_BOS_ONLY_NPU_OK: graph=qnn_partition_2 outputs=30"
             <<" execute_ms="<<ms<<" baseline_buffers_unchanged=1\n"<<std::flush;

    struct Stats {size_t bytes=0,ab=0,ac=0,bc=0;
                  double real_ab=0,real_ac=0,real_bc=0;};
    Stats total;
    std::array<std::array<Stats,2>,2> groups{};
    const auto signed_q=[](uint8_t byte)->int {
        return byte<128?static_cast<int>(byte):static_cast<int>(byte)-256;
    };
    const auto accumulate=[](Stats& dst,const Stats& source) {
        dst.bytes+=source.bytes;dst.ab+=source.ab;
        dst.ac+=source.ac;dst.bc+=source.bc;
        dst.real_ab+=source.real_ab;dst.real_ac+=source.real_ac;
        dst.real_bc+=source.real_bc;
    };
    for(size_t n=0;n<links.size();++n) {
        const auto& link=links[n];
        const std::string name="StatefulPartitionedCall_1_"+std::to_string(n);
        size_t index=prefill.outputs.size();
        for(size_t j=0;j<prefill.outputs.size();++j) {
            const char* found=tensor_name(prefill.outputs[j]);
            if(found && name==found) {
                if(index!=prefill.outputs.size())
                    throw std::runtime_error("STAGE6_12_DUPLICATE_PREFILL_SOURCE");
                index=j;
            }
        }
        if(index==prefill.outputs.size() ||
           tensor_dtype(prefill.outputs[index])!=QNN_DATATYPE_SFIXED_POINT_8 ||
           fresh[index].size()!=static_cast<size_t>(link.channels)*128 ||
           saved_prefill[index].size()!=fresh[index].size() ||
           saved_decode.at(link.output_index).size()!=link.channels)
            throw std::runtime_error("STAGE6_12_BAD_OUTPUT: "+name);
        const auto& quant=tensor_quant(prefill.outputs[index]);
        const auto& decode_quant=tensor_quant(decode.outputs[link.output_index]);
        if(quant.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           decode_quant.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           quant.scaleOffsetEncoding.scale!=decode_quant.scaleOffsetEncoding.scale ||
           quant.scaleOffsetEncoding.offset!=decode_quant.scaleOffsetEncoding.offset)
            throw std::runtime_error("STAGE6_12_OUTPUT_QUANT_MISMATCH: "+name);
        const double scale=quant.scaleOffsetEncoding.scale;
        Stats stats; stats.bytes=link.channels;
        for(size_t c=0;c<link.channels;++c) {
            const size_t offset=link.key?c*128:c; // K: [ch,128], V: [128,ch]
            const int A=signed_q(fresh[index][offset]);
            const int B=signed_q(saved_prefill[index][offset]);
            const int C=signed_q(saved_decode[link.output_index][c]);
            stats.ab+=A!=B;stats.ac+=A!=C;stats.bc+=B!=C;
            stats.real_ab+=std::abs(A-B)*scale;
            stats.real_ac+=std::abs(A-C)*scale;
            stats.real_bc+=std::abs(B-C)*scale;
        }
        accumulate(total,stats);
        accumulate(groups[link.key?0:1][link.channels==256?0:1],stats);
        std::ostringstream record;
        record<<std::setprecision(9)<<"PREFILL_BOS_ONLY_KV_TENSOR: name="<<name
              <<" kind="<<(link.key?"K":"V")
              <<" channels="<<link.channels
              <<" AB_mismatches="<<stats.ab
              <<" AC_mismatches="<<stats.ac
              <<" BC_mismatches="<<stats.bc
              <<" real_mae_AB="<<(stats.real_ab/stats.bytes)
              <<" real_mae_AC="<<(stats.real_ac/stats.bytes)
              <<" real_mae_BC="<<(stats.real_bc/stats.bytes);
        std::cout<<record.str()<<"\n"<<std::flush;
    }
    if(total.bytes!=9216)throw std::runtime_error("STAGE6_12_TOTAL_NOT_9216");
    for(size_t kind=0;kind<2;++kind)
        for(size_t width=0;width<2;++width) {
            const auto& stats=groups[kind][width];
            if(!stats.bytes)throw std::runtime_error("STAGE6_12_EMPTY_KV_GROUP");
            std::ostringstream record;
            record<<std::setprecision(9)
                  <<"PREFILL_BOS_ONLY_KV_GROUP: kind="<<(kind?"V":"K")
                  <<" channels="<<(width?512:256)
                  <<" bytes="<<stats.bytes
                  <<" AB_mismatches="<<stats.ab
                  <<" AC_mismatches="<<stats.ac
                  <<" BC_mismatches="<<stats.bc
                  <<" real_mae_AB="<<(stats.real_ab/stats.bytes)
                  <<" real_mae_AC="<<(stats.real_ac/stats.bytes)
                  <<" real_mae_BC="<<(stats.real_bc/stats.bytes);
            std::cout<<record.str()<<"\n"<<std::flush;
        }
    std::cout<<"PREFILL_BOS_ONLY_KV_TOTAL: tensors=30 bytes="<<total.bytes
             <<" AB_mismatches="<<total.ab
             <<" AC_mismatches="<<total.ac
             <<" BC_mismatches="<<total.bc
             <<" AB_bit_exact="<<(total.ab==0?1:0)
             <<" no_reference_mutation=1\n"<<std::flush;
    std::cout<<"PREFILL_BOS_ONLY_STAGE6_12_OK: diagnostic_complete=1"
             <<" numerical_equivalence_not_assumed=1\n"<<std::flush;
}
