// Stage 6.14: Decode-only 17-token prompt control.
// It starts AFTER Stage 6.11 zero-KV Decode(pos0, BOS=2) has executed,
// then feeds its K/V slices into 30 fresh scratch caches, executes pos1..16,
// and runs LM Head on the final hidden state. No Prefill outputs are used
// to construct the cache. Prefill KV rows are read ONLY for diagnostics.
void run_stage6_14_decode_only(
    QnnRuntime& runtime,const ReplayJob& prefill,ReplayJob& decode,
    ReplayJob& head,size_t hidden_index,
    const std::vector<KvLink>& links,const Options& options,
    std::vector<std::vector<uint8_t>>& scratch) {
    if(prefill.graph_name!="qnn_partition_2" || prefill.outputs.size()!=30 ||
       decode.graph_name!="qnn_partition_0" || decode.inputs.size()!=38 ||
       decode.outputs.size()!=31 || head.graph_name!="qnn_partition_1" ||
       links.size()!=30 || scratch.size()!=38 ||
       options.prefill_kv_steps!=1 || !options.prefill_seed_first_prompt)
        throw std::runtime_error("DECODE_ONLY_PREREQUISITES_INVALID");
    const auto tokens=read_file(options.cpu_prefill_tokens);
    if(tokens.size()!=512)throw std::runtime_error("DECODE_ONLY_PROMPT_BYTES_INVALID");
    std::array<int32_t,128> prompt{};
    std::memcpy(prompt.data(),tokens.data(),tokens.size());
    size_t valid=0;
    while(valid<128 && prompt[valid]!=0)++valid;
    if(valid!=17 || prompt[0]!=2)
        throw std::runtime_error("DECODE_ONLY_EXPECTED_17_TOKEN_BOS_PROMPT");
    for(size_t pos=valid;pos<128;++pos)
        if(prompt[pos]!=0)
            throw std::runtime_error("DECODE_ONLY_NONCONTIGUOUS_PADDING");
    std::set<size_t> kv_indices;
    for(const auto& link:links) {
        if(!kv_indices.insert(link.input_index).second ||
           scratch.at(link.input_index).size()!=
               tensor_nbytes(decode.inputs.at(link.input_index)) ||
           std::any_of(scratch[link.input_index].begin(),
                       scratch[link.input_index].end(),
                       [](uint8_t byte){return byte!=0;}))
            throw std::runtime_error("DECODE_ONLY_INITIAL_CACHE_NOT_ZERO");
        const auto& q=tensor_quant(decode.inputs[link.input_index]);
        if(q.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           q.scaleOffsetEncoding.offset!=0)
            throw std::runtime_error("DECODE_ONLY_CACHE_QUANT_ZERO_UNSUPPORTED");
    }
    std::cout<<"DECODE_ONLY_BEGIN: prompt_tokens=17"
             <<" first_token=2 first_decode_already_executed=1"
             <<" zero_initial_KV=1"
             <<" KV_from_Decode_only=1 Prefill_KV_used_for_inputs=0"
             <<" saved_reference_inputs_used=0\n"<<std::flush;

    // Source name comes from verified Stage 6.10/6.11 KV mapping, not
    // graph output order. Read-only Prefill outputs never feed Decode.
    std::vector<size_t> source_indices;
    source_indices.reserve(30);
    for(size_t n=0;n<links.size();++n) {
        const std::string wanted="StatefulPartitionedCall_1_"+std::to_string(n);
        size_t match=prefill.outputs.size();
        for(size_t i=0;i<prefill.outputs.size();++i) {
            const char* actual=tensor_name(prefill.outputs[i]);
            if(actual && wanted==actual) {
                if(match!=prefill.outputs.size())
                    throw std::runtime_error("DECODE_ONLY_AMBIGUOUS_PREFILL_KV");
                match=i;
            }
        }
        if(match==prefill.outputs.size())
            throw std::runtime_error("DECODE_ONLY_PREFILL_KV_NOT_FOUND: "+wanted);
        const auto& q=tensor_quant(prefill.outputs[match]);
        const auto& d=tensor_quant(decode.outputs[links[n].output_index]);
        if(tensor_dtype(prefill.outputs[match])!=QNN_DATATYPE_SFIXED_POINT_8 ||
           q.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           d.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           q.scaleOffsetEncoding.scale!=d.scaleOffsetEncoding.scale ||
           q.scaleOffsetEncoding.offset!=d.scaleOffsetEncoding.offset ||
           prefill.output_buffers[match].size()!=size_t(links[n].channels)*128)
            throw std::runtime_error("DECODE_ONLY_KV_QUANT_OR_SHAPE_MISMATCH");
        source_indices.push_back(match);
    }
    const auto verified_prefill=prefill.output_buffers;

    const auto signed_q=[](uint8_t b)->int {
        return b<128?int(b):int(b)-256;
    };
    const auto compare_position=[&](size_t pos) {
        size_t diff=0,bytes=0;
        std::array<size_t,4> group_diff{};
        std::array<size_t,4> group_bytes{};
        double real_absolute=0;
        for(size_t n=0;n<links.size();++n) {
            const auto& link=links[n];
            const auto& source=verified_prefill[source_indices[n]];
            const auto& slice=decode.output_buffers.at(link.output_index);
            const double scale=tensor_quant(
                prefill.outputs[source_indices[n]]).scaleOffsetEncoding.scale;
            if(slice.size()!=link.channels)
                throw std::runtime_error("DECODE_ONLY_BAD_SLICE_SIZE");
            const size_t group=(link.key?0:2)+(link.channels==256?0:1);
            for(size_t c=0;c<link.channels;++c) {
                const size_t offset=link.key?c*128+pos:pos*link.channels+c;
                const int actual=signed_q(slice[c]);
                const int expected=signed_q(source[offset]);
                const size_t different=static_cast<size_t>(actual!=expected);
                diff+=different;group_diff[group]+=different;
                real_absolute+=std::abs(actual-expected)*scale;
            }
            bytes+=link.channels;
            group_bytes[group]+=link.channels;
        }
        if(bytes!=9216)
            throw std::runtime_error("DECODE_ONLY_COMPARE_BYTES_INVALID");
        std::ostringstream msg;
        msg<<std::setprecision(9)
           <<"DECODE_ONLY_KV_POSITION: pos="<<pos
           <<" token_id="<<prompt[pos]
           <<" compared_bytes="<<bytes
           <<" differing_bytes="<<diff
           <<" K256="<<group_diff[0]<<"/"<<group_bytes[0]
           <<" K512="<<group_diff[1]<<"/"<<group_bytes[1]
           <<" V256="<<group_diff[2]<<"/"<<group_bytes[2]
           <<" V512="<<group_diff[3]<<"/"<<group_bytes[3]
           <<" real_mae_all="<<(real_absolute/bytes)
           <<" prefill_KV_comparison_only=1";
        std::cout<<msg.str()<<"\n"<<std::flush;
    };

    auto inputs=decode.inputs;
    const auto original_decode_inputs=decode.input_buffers;
    const auto set_nonk=[&](const std::string& name,Qnn_DataType_t dtype,
                            size_t bytes,std::vector<uint8_t> raw) {
        size_t index=inputs.size();
        for(size_t i=0;i<inputs.size();++i) {
            const char* actual=tensor_name(inputs[i]);
            if(actual && name==actual) {
                if(index!=inputs.size())
                    throw std::runtime_error("DECODE_ONLY_DUPLICATE_NONKV");
                index=i;
            }
        }
        if(index==inputs.size() || kv_indices.count(index) ||
           tensor_dtype(inputs[index])!=dtype ||
           tensor_nbytes(inputs[index])!=bytes || raw.size()!=bytes)
            throw std::runtime_error("DECODE_ONLY_BAD_NONKV: "+name);
        scratch[index]=std::move(raw);
        bind_raw(inputs[index],scratch[index]);
    };
    const std::array<size_t,6> aux_bytes{512,512,256,256,4097,4097};

    // Already executed Decode(pos0) is preserved as the first step:
    // its 30 K/V slices are still in decode.output_buffers.
    for(size_t pos=0;pos<valid;++pos) {
        if(pos>0) {
            set_nonk("decode_embeddings_0",QNN_DATATYPE_FLOAT_32,6144,
                gemma4::compute_packed_embedding(
                    options.cpu_embed_section0,0,prompt[pos]));
            set_nonk("decode_per_layer_embeddings_0",QNN_DATATYPE_FLOAT_32,35840,
                gemma4::compute_packed_embedding(
                    options.cpu_embed_section2,2,prompt[pos]));
            auto aux=gemma4::run_decode_aux(
                options.cpu_aux_model,static_cast<int32_t>(pos),prompt[pos]);
            if(aux.size()!=6)
                throw std::runtime_error("DECODE_ONLY_AUX_NOT_SIX");
            for(size_t j=0;j<aux.size();++j)
                set_nonk(aux[j].qnn_name,QNN_DATATYPE_SFIXED_POINT_8,
                         aux_bytes[j],std::move(aux[j].bytes));
            // Existing C++ buffers are independent of the saved Decode
            // input samples. Rebind all because vector moves change pointers.
            for(size_t i=0;i<inputs.size();++i) {
                if(scratch[i].size()!=tensor_nbytes(inputs[i]))
                    throw std::runtime_error("DECODE_ONLY_INPUT_SIZE_CHANGED");
                bind_raw(inputs[i],scratch[i]);
            }
            const auto begin=std::chrono::steady_clock::now();
            runtime.execute(decode.handle,inputs,decode.outputs);
            const double elapsed=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-begin).count();
            std::cout<<"DECODE_ONLY_STEP_OK: pos="<<pos
                     <<" token_id="<<prompt[pos]
                     <<" NPU_decode_ms="<<elapsed
                     <<" saved_decode_reference_claim=0\n"<<std::flush;
        }
        compare_position(pos);
        size_t written=0;
        for(const auto& link:links) {
            const auto& slice=decode.output_buffers.at(link.output_index);
            auto& cache=scratch.at(link.input_index);
            gemma4::update_kv_slot(cache,slice,link.channels,4096,pos,
                link.key?gemma4::KvLayout::Key:gemma4::KvLayout::Value);
            for(size_t c=0;c<link.channels;++c) {
                const size_t offset=link.key?
                    c*4096+pos:pos*link.channels+c;
                if(cache[offset]!=slice[c])
                    throw std::runtime_error("DECODE_ONLY_KV_FEEDBACK_MISMATCH");
            }
            written+=slice.size();
        }
        if(written!=9216)
            throw std::runtime_error("DECODE_ONLY_INCORRECT_FEEDBACK_SIZE");
    }
    if(prefill.output_buffers!=verified_prefill ||
       decode.input_buffers!=original_decode_inputs)
        throw std::runtime_error("DECODE_ONLY_MODIFIED_REFERENCE_BUFFERS");
    std::cout<<"DECODE_ONLY_PROMPT_KV_READY: positions=17 cached_slices=510"
             <<" last_prompt_token="<<prompt[valid-1]
             <<" KV_from_Decode_only=1 Prefill_KV_used_for_inputs=0"
             <<" reference_buffers_unchanged=1\n"<<std::flush;

    // Final hidden state belongs to the LAST real prompt token, pos16.
    execute_chained_decode_head(runtime,decode,head,hidden_index);
    const auto& logits=head.output_buffers.at(0);
    if(logits.size()!=262144 ||
       tensor_dtype(head.outputs.at(0))!=QNN_DATATYPE_SFIXED_POINT_8)
        throw std::runtime_error("DECODE_ONLY_HEAD_BAD_LOGITS");
    int best=-129;
    std::vector<int32_t> tied;
    for(size_t i=0;i<logits.size();++i) {
        const int q=signed_q(logits[i]);
        if(q>best){best=q;tied.clear();tied.push_back(static_cast<int32_t>(i));}
        else if(q==best)tied.push_back(static_cast<int32_t>(i));
    }
    if(tied.empty())
        throw std::runtime_error("DECODE_ONLY_HEAD_NO_MAX");
    std::ostringstream candidates;
    for(size_t i=0;i<tied.size() && i<32;++i)
        candidates<<(i?",":"")<<tied[i];
    if(tied.size()>32)candidates<<",...(truncated)";
    std::cout<<"DECODE_ONLY_FIRST_GENERATION_CANDIDATES:"
             <<" prompt_tokens=17 next_position=17"
             <<" max_logit_q="<<best
             <<" tied_max_tokens="<<tied.size()
             <<" candidate_ids="<<candidates.str()
             <<" int8_logits_only=1 no_fp32_ranking_claim=1\n"<<std::flush;
    std::cout<<"DECODE_ONLY_STAGE6_14_OK: steps=17"
             <<" all_nonKV_from_CPU=1 all_KV_from_Decode_only=1"
             <<" prefill_KV_read_only=1 ready_for_next_generation_step=1"
             <<" no_semantic_equivalence_claim=1\n"<<std::flush;
}
