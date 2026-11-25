// Stage 6.15: controlled same-graph experiment at the LAST prompt token.
// Both branches use the SAME Decode(pos16, token4368), same eight CPU
// non-KV tensors and same LM Head. The sole difference is K/V history:
//   D: KV from Decode-only pos0..15, already executed in Stage 6.14.
//   P: KV from Prefill output rows 0..15, copied to separate scratch RAM.
// No input/saved reference/Decode-only output is mutated.
void run_stage6_15_compare_kv_histories(
    QnnRuntime& runtime,const ReplayJob& prefill,ReplayJob& decode,
    ReplayJob& head,size_t hidden_index,
    const std::vector<KvLink>& links,
    const std::vector<std::vector<uint8_t>>& decoded_inputs,
    const std::array<int32_t,128>& prompt) {
    if(prefill.outputs.size()!=30 || decode.inputs.size()!=38 ||
       decode.outputs.size()!=31 || head.outputs.size()!=1 ||
       links.size()!=30 || hidden_index>=decode.outputs.size() ||
       prompt[16]!=4368)
        throw std::runtime_error("KV_HISTORY_METADATA_OR_TOKEN_INVALID");
    const auto saved_decode_outputs=decode.output_buffers;
    const auto saved_head_logits=head.output_buffers.at(0);
    const auto saved_prefill_outputs=prefill.output_buffers;
    const auto saved_decode_input_refs=decode.input_buffers;
    if(saved_head_logits.size()!=262144 ||
       saved_decode_outputs.at(hidden_index).size()!=1536)
        throw std::runtime_error("KV_HISTORY_DECODE_ONLY_OUTPUT_INVALID");
    if(decoded_inputs.size()!=decode.inputs.size())
        throw std::runtime_error("KV_HISTORY_DECODE_ONLY_INPUT_COUNT_INVALID");

    // Reconstruct P's K/V prefix, without touching D's caches or pos16.
    auto alternate_inputs=decoded_inputs;
    std::set<size_t> kv_input_indices;
    size_t copied_bytes=0;
    size_t differing_cache_bytes=0;
    constexpr size_t kCapacity=4096,kPrefillRows=128,kPrefixRows=16;
    for(size_t n=0;n<links.size();++n) {
        const auto& link=links[n];
        if(!kv_input_indices.insert(link.input_index).second ||
           alternate_inputs.at(link.input_index).size()!=
               tensor_nbytes(decode.inputs.at(link.input_index)))
            throw std::runtime_error("KV_HISTORY_CACHE_INPUT_BAD");
        const std::string source_name="StatefulPartitionedCall_1_"+
            std::to_string(n);
        size_t source_index=prefill.outputs.size();
        for(size_t i=0;i<prefill.outputs.size();++i) {
            const char* name=tensor_name(prefill.outputs[i]);
            if(name && source_name==name) {
                if(source_index!=prefill.outputs.size())
                    throw std::runtime_error("KV_HISTORY_AMBIGUOUS_PREFILL_SOURCE");
                source_index=i;
            }
        }
        if(source_index==prefill.outputs.size())
            throw std::runtime_error("KV_HISTORY_PREFILL_SOURCE_MISSING");
        const auto& src=prefill.output_buffers.at(source_index);
        const auto& qs=tensor_quant(prefill.outputs.at(source_index));
        const auto& qd=tensor_quant(decode.inputs.at(link.input_index));
        if(src.size()!=size_t(link.channels)*kPrefillRows ||
           qs.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           qd.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           qs.scaleOffsetEncoding.scale!=qd.scaleOffsetEncoding.scale ||
           qs.scaleOffsetEncoding.offset!=qd.scaleOffsetEncoding.offset ||
           qd.scaleOffsetEncoding.offset!=0)
            throw std::runtime_error("KV_HISTORY_PREFILL_CACHE_QUANT_MISMATCH");

        // P starts completely empty and receives 16 authentic Prefill rows.
        auto& prefix=alternate_inputs[link.input_index];
        prefix.assign(prefix.size(),uint8_t{0});
        gemma4::update_kv_block(prefix,src,link.channels,kCapacity,0,
            kPrefillRows,kPrefixRows,
            link.key?gemma4::KvLayout::Key:gemma4::KvLayout::Value);
        for(size_t row=0;row<kPrefixRows;++row)
            for(size_t c=0;c<link.channels;++c) {
                const size_t off=link.key?c*kCapacity+row:row*link.channels+c;
                differing_cache_bytes+=
                    static_cast<size_t>(
                        prefix[off]!=decoded_inputs[link.input_index][off]);
                ++copied_bytes;
            }
        // No unused future/cache slots copied from Prefill.
        const size_t test_slot=link.key?16:16*link.channels;
        if(prefix.at(test_slot)!=0)
            throw std::runtime_error("KV_HISTORY_POS16_NOT_EMPTY");
    }
    if(kv_input_indices.size()!=30 || copied_bytes!=9216*kPrefixRows)
        throw std::runtime_error("KV_HISTORY_PREFIX_COUNT_INVALID");

    // Inputs 8/38 are identical by construction. Verify them byte-for-byte.
    size_t fixed_nonkv=0,fixed_bytes=0;
    for(size_t i=0;i<decode.inputs.size();++i) {
        if(kv_input_indices.count(i))continue;
        if(decoded_inputs[i].size()!=tensor_nbytes(decode.inputs[i]) ||
           alternate_inputs[i]!=decoded_inputs[i])
            throw std::runtime_error("KV_HISTORY_NONKV_CHANGED");
        ++fixed_nonkv;
        fixed_bytes+=alternate_inputs[i].size();
    }
    if(fixed_nonkv!=8 || fixed_bytes!=51714)
        throw std::runtime_error("KV_HISTORY_EIGHT_NONKV_INVALID");
    std::cout<<"KV_HISTORY_CONTROL_BEGIN: pos=16 token_id=4368"
             <<" prefix_tokens=16"
             <<" same_decode_graph=1 same_head_graph=1"
             <<" same_nonk_inputs=8 same_nonk_bytes="<<fixed_bytes
             <<" differing_prefix_KV_bytes="<<differing_cache_bytes
             <<" compared_prefix_KV_bytes="<<copied_bytes
             <<" only_history_changes=1\n"<<std::flush;

    // QNN descriptors reference temporary storage for both inputs and outputs.
    // Baseline output buffers must remain intact for the full comparison.
    auto qnn_inputs=decode.inputs;
    for(size_t i=0;i<qnn_inputs.size();++i)
        bind_raw(qnn_inputs[i],alternate_inputs[i]);
    auto qnn_outputs=decode.outputs;
    std::vector<std::vector<uint8_t>> alternate_outputs(qnn_outputs.size());
    for(size_t i=0;i<qnn_outputs.size();++i) {
        alternate_outputs[i].resize(tensor_nbytes(qnn_outputs[i]));
        bind_raw(qnn_outputs[i],alternate_outputs[i]);
    }
    const auto t=std::chrono::steady_clock::now();
    runtime.execute(decode.handle,qnn_inputs,qnn_outputs);
    const double decode_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t).count();
    std::cout<<"KV_HISTORY_PREFILL_PREFIX_DECODE_OK: pos=16 token_id=4368"
             <<" NPU_decode_ms="<<decode_ms
             <<" output_buffers_isolated=1\n"<<std::flush;

    auto head_inputs=head.inputs;
    auto head_outputs=head.outputs;
    if(head_inputs.size()!=1 || head_outputs.size()!=1 ||
       tensor_nbytes(head_inputs[0])!=alternate_outputs[hidden_index].size())
        throw std::runtime_error("KV_HISTORY_HEAD_METADATA_MISMATCH");
    bind_raw(head_inputs[0],alternate_outputs[hidden_index]);
    std::vector<uint8_t> alternate_logits(tensor_nbytes(head_outputs[0]));
    bind_raw(head_outputs[0],alternate_logits);
    runtime.execute(head.handle,head_inputs,head_outputs);
    if(alternate_logits.size()!=262144)
        throw std::runtime_error("KV_HISTORY_LOGITS_COUNT_INVALID");
    std::cout<<"KV_HISTORY_PREFILL_PREFIX_HEAD_OK: logits_bytes=262144"
             <<" independent_head_output_buffer=1\n"<<std::flush;

    if(decode.output_buffers!=saved_decode_outputs ||
       head.output_buffers[0]!=saved_head_logits ||
       prefill.output_buffers!=saved_prefill_outputs ||
       decode.input_buffers!=saved_decode_input_refs)
        throw std::runtime_error("KV_HISTORY_MUTATED_BASELINE_OUTPUTS");

    const auto signed_q=[](uint8_t b)->int {
        return b<128?int(b):int(b)-256;
    };
    const auto summarize=[&](const std::vector<uint8_t>& logits,
                             const char* path) {
        int best=-129;
        std::vector<size_t> ids;
        for(size_t i=0;i<logits.size();++i) {
            const int q=signed_q(logits[i]);
            if(q>best) {best=q;ids.clear();ids.push_back(i);}
            else if(q==best)ids.push_back(i);
        }
        if(ids.empty())throw std::runtime_error("KV_HISTORY_LOGIT_MAX_MISSING");
        std::ostringstream out;
        for(size_t i=0;i<ids.size() && i<24;++i)
            out<<(i?",":"")<<ids[i];
        if(ids.size()>24)out<<",...(truncated)";
        std::cout<<"KV_HISTORY_LOGIT_MAX: path="<<path
                 <<" max_q="<<best
                 <<" ties="<<ids.size()
                 <<" candidate_ids="<<out.str()
                 <<" no_fp32_ranking_claim=1\n"<<std::flush;
        return ids;
    };
    const auto idsD=summarize(saved_head_logits,"decode-only");
    const auto idsP=summarize(alternate_logits,"prefill-prefix");

    size_t changed_hidden=0,changed_logits=0;
    uint64_t sum_hidden_delta=0,sum_logit_delta=0;
    for(size_t i=0;i<saved_decode_outputs[hidden_index].size();++i) {
        const int a=signed_q(saved_decode_outputs[hidden_index][i]);
        const int b=signed_q(alternate_outputs[hidden_index][i]);
        changed_hidden+=static_cast<size_t>(a!=b);
        sum_hidden_delta+=static_cast<uint64_t>(std::abs(a-b));
    }
    for(size_t i=0;i<saved_head_logits.size();++i) {
        const int a=signed_q(saved_head_logits[i]);
        const int b=signed_q(alternate_logits[i]);
        changed_logits+=static_cast<size_t>(a!=b);
        sum_logit_delta+=static_cast<uint64_t>(std::abs(a-b));
    }
    size_t shared_top=0;
    for(const size_t id:idsD)
        shared_top+=static_cast<size_t>(
            std::find(idsP.begin(),idsP.end(),id)!=idsP.end());
    size_t diff_kv=0;
    for(const auto& link:links) {
        const auto& a=saved_decode_outputs.at(link.output_index);
        const auto& b=alternate_outputs.at(link.output_index);
        if(a.size()!=link.channels || b.size()!=a.size())
            throw std::runtime_error("KV_HISTORY_SLICE_MISMATCH");
        for(size_t i=0;i<a.size();++i)
            diff_kv+=static_cast<size_t>(a[i]!=b[i]);
    }
    std::ostringstream report;
    report<<std::setprecision(9)
          <<"KV_HISTORY_COMPARE: pos=16 token_id=4368"
          <<" prefix_KV_diff="<<differing_cache_bytes<<"/"<<copied_bytes
          <<" output_KV_diff="<<diff_kv<<"/9216"
          <<" hidden_diff="<<changed_hidden<<"/1536"
          <<" logits_diff="<<changed_logits<<"/262144"
          <<" avg_abs_hidden_delta="<<double(sum_hidden_delta)/1536
          <<" avg_abs_logits_delta="<<double(sum_logit_delta)/262144
          <<" shared_max_ids="<<shared_top
          <<" exact_hidden="<<(changed_hidden==0?1:0)
          <<" exact_logits="<<(changed_logits==0?1:0)
          <<" same_graph_and_nonKV=1";
    std::cout<<report.str()<<"\n"<<std::flush;
    std::cout<<"KV_HISTORY_STAGE6_15_OK: read_only=1"
             <<" Decode_only_and_Prefill_prefix_compared=1"
             <<" numerical_equivalence_not_assumed=1\n"<<std::flush;
}
