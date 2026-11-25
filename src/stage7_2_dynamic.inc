// Stage 7.2: metadata-only, sample-free Prefill->Decode->LM Head path.
// The manifest supplies ONLY the three AOT binary paths and graph names.
// Input-list and Result_0 fields are deliberately NOT OPENED in this mode.
// At no point are saved QNN sample inputs or references used for inference.
void run_stage7_2_dynamic(
    std::vector<std::unique_ptr<ReplayJob>>& jobs,const Options& options,
    const std::chrono::steady_clock::time_point& bench8_job_started) {
    if(jobs.size()!=3 || jobs.front()->graph_name!="qnn_partition_0")
        throw std::runtime_error("DYNAMIC_REQUIRES_DECODE_FIRST_THREE_GRAPH_MANIFEST");
    ReplayJob *decode=nullptr,*prefill=nullptr,*head=nullptr;
    for(auto& j:jobs) {
        if(j->graph_name=="qnn_partition_0") {
            if(decode)throw std::runtime_error("DYNAMIC_DUPLICATE_DECODE");
            decode=j.get();
        } else if(j->graph_name=="qnn_partition_2") {
            if(prefill)throw std::runtime_error("DYNAMIC_DUPLICATE_PREFILL");
            prefill=j.get();
        } else if(j->graph_name=="qnn_partition_1") {
            if(head)throw std::runtime_error("DYNAMIC_DUPLICATE_HEAD");
            head=j.get();
        } else throw std::runtime_error("DYNAMIC_UNKNOWN_GRAPH");
    }
    if(!decode || !prefill || !head ||
       decode->inputs.size()!=38 || decode->outputs.size()!=31 ||
       prefill->inputs.size()!=36 || prefill->outputs.size()!=30 ||
       head->inputs.size()!=1 || head->outputs.size()!=1)
        throw std::runtime_error("DYNAMIC_GRAPH_METADATA_INVALID");
    const size_t hidden=find_decode_head_link(*decode,*head);
    const auto raw=read_file(options.dynamic_prompt_tokens);
    if(raw.size()!=128*sizeof(int32_t))
        throw std::runtime_error("DYNAMIC_PROMPT_NEEDS_128_INT32");
    std::vector<int32_t> prompt(128);
    std::memcpy(prompt.data(),raw.data(),raw.size());
    size_t n=0;
    while(n<128 && prompt[n]!=0)++n;
    if(n<8 || n>=128 || prompt[0]!=2)
        throw std::runtime_error("DYNAMIC_PROMPT_LENGTH_OR_BOS_INVALID");
    for(size_t i=0;i<n;++i)
        if(prompt[i]<=0 || prompt[i]>=262144)
            throw std::runtime_error("DYNAMIC_PROMPT_ID_OUT_OF_RANGE");
    for(size_t i=n;i<128;++i)
        if(prompt[i]!=0)
            throw std::runtime_error("DYNAMIC_PROMPT_PADDING_NONCONTIGUOUS");
    // The BPE encoder emits [..user ID..,turn_end,LF,turn,model,LF].
    // Our verified first-token path replays that last LF with Decode,
    // using Prefill KV rows before LF. This is Stage6.18's exact algorithm.
    if(prompt[n-5]!=106 || prompt[n-4]!=107 ||
       prompt[n-3]!=105 || prompt[n-2]!=4368 || prompt[n-1]!=107)
        throw std::runtime_error("DYNAMIC_PROMPT_CHAT_TEMPLATE_INVALID");
    const int32_t seed=prompt[n-1];
    const size_t seed_pos=n-1;
    prompt[seed_pos]=0; // RAM ONLY; the new on-disk RAW stays untouched.
    std::cout<<"DYNAMIC_PROMPT_READY: chat_tokens="<<n
             <<" prefill_tokens="<<seed_pos
             <<" decode_seed_position="<<seed_pos
             <<" decode_seed_token="<<seed
             <<" padding_tokens="<<(128-seed_pos)
             <<" samples_read=0 references_read=0\n"<<std::flush;

    // Metadata-validated CPU Prefill inputs and 28 zeroed KV tensors.
    const auto bench8_cpu_prefill_start=std::chrono::steady_clock::now();
    std::set<size_t> prefill_nonkv;
    auto install_prefill=[&](size_t index,const char* name,
                             Qnn_DataType_t dtype,size_t bytes,
                             std::vector<uint8_t> value) {
        if(index>=prefill->inputs.size() || !prefill_nonkv.insert(index).second ||
           !tensor_name(prefill->inputs[index]) ||
           std::string(tensor_name(prefill->inputs[index]))!=name ||
           tensor_dtype(prefill->inputs[index])!=dtype ||
           tensor_nbytes(prefill->inputs[index])!=bytes ||
           value.size()!=bytes)
            throw std::runtime_error(std::string("DYNAMIC_PREFILL_INPUT_BAD: ")+name);
        prefill->input_buffers[index]=std::move(value);
    };
    install_prefill(32,"prefill_128_embeddings_0",
        QNN_DATATYPE_FLOAT_32,786432,
        gemma4::compute_packed_prefill_embeddings(
            options.dynamic_section0,0,prompt));
    install_prefill(7,"prefill_128_per_layer_embeddings_0",
        QNN_DATATYPE_FLOAT_32,4587520,
        gemma4::compute_packed_prefill_embeddings(
            options.dynamic_section2,2,prompt));
    auto aux=gemma4::run_prefill_aux(options.dynamic_section1,prompt);
    if(aux.size()!=6)
        throw std::runtime_error("DYNAMIC_PREFILL_AUX_NOT_SIX");
    struct PrefillAux {size_t index;const char* name;size_t bytes;};
    const std::array<PrefillAux,6> aux_specs{{
        {3,"prefill_128_pos_emb_cos_0",65536},
        {5,"prefill_128_pos_emb_sin_0",65536},
        {6,"prefill_128_pos_emb_local_sin_0",32768},
        {18,"prefill_128_pos_emb_local_cos_0",32768},
        {21,"prefill_128_mask_local_0",540672},
        {26,"prefill_128_mask_global_0",540672}
    }};
    for(size_t i=0;i<aux_specs.size();++i) {
        if(aux[i].qnn_name!=aux_specs[i].name)
            throw std::runtime_error("DYNAMIC_PREFILL_AUX_ORDER_UNEXPECTED");
        install_prefill(aux_specs[i].index,aux_specs[i].name,
                        QNN_DATATYPE_SFIXED_POINT_8,aux_specs[i].bytes,
                        std::move(aux[i].bytes));
    }
    size_t zero_kv_count=0;
    for(size_t i=0;i<prefill->inputs.size();++i) {
        if(prefill_nonkv.count(i))continue;
        const auto& tensor=prefill->inputs[i];
        const char* name=tensor_name(tensor);
        const auto& q=tensor_quant(tensor);
        if(!name || std::string(name).find("prefill_128_kv_cache_")!=0 ||
           tensor_dtype(tensor)!=QNN_DATATYPE_SFIXED_POINT_8 ||
           q.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           q.scaleOffsetEncoding.offset!=0 ||
           !std::all_of(prefill->input_buffers[i].begin(),
                        prefill->input_buffers[i].end(),
                        [](uint8_t b){return b==0;}))
            throw std::runtime_error("DYNAMIC_PREFILL_KV_ZERO_UNSUPPORTED");
        ++zero_kv_count;
    }
    if(prefill_nonkv.size()!=8 || zero_kv_count!=28)
        throw std::runtime_error("DYNAMIC_PREFILL_EXPECTED_8_CPU_28_KV_INPUTS");
    for(size_t i=0;i<prefill->inputs.size();++i)
        bind_raw(prefill->inputs[i],prefill->input_buffers[i]);
    std::cout<<"DYNAMIC_PREFILL_INPUTS_READY: from_CPU=8"
             <<" zero_KV=28 no_saved_QNN_samples=1\n"<<std::flush;
    const auto bench8_cpu_prefill_done=std::chrono::steady_clock::now();

    const auto links=decode_kv_links(*decode);
    const auto bench8_context_start=std::chrono::steady_clock::now();
    QnnRuntime runtime(options.backend);
    for(auto& j:jobs) {
        const auto start=std::chrono::steady_clock::now();
        j->handle=runtime.load(j->binary,j->graph_name);
        const auto ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
        std::cout<<"DYNAMIC_CONTEXT_READY: graph="<<j->graph_name
                 <<" load_ms="<<ms<<"\n"<<std::flush;
    }
    const auto bench8_context_done=std::chrono::steady_clock::now();
    const auto bench8_prefill_npu_start=std::chrono::steady_clock::now();
    runtime.execute(prefill->handle,prefill->inputs,prefill->outputs);
    const auto bench8_prefill_npu_done=std::chrono::steady_clock::now();
    std::cout<<"DYNAMIC_PREFILL_NPU_OK: inputs=36 outputs=30"
             <<" valid_rows="<<seed_pos<<" padding_ignored=1\n"<<std::flush;

    const auto bench8_kv_bridge_start=std::chrono::steady_clock::now();
    size_t prefill_sources=0,cache_bytes=0;
    std::set<size_t> kv_indexes;
    for(size_t index=0;index<links.size();++index) {
        const auto& link=links[index];
        if(!kv_indexes.insert(link.input_index).second)
            throw std::runtime_error("DYNAMIC_KV_INPUT_DUPLICATE");
        const std::string source_name="StatefulPartitionedCall_1_"+
            std::to_string(index);
        size_t found=prefill->outputs.size();
        for(size_t j=0;j<prefill->outputs.size();++j) {
            const char* name=tensor_name(prefill->outputs[j]);
            if(name && source_name==name) {
                if(found!=prefill->outputs.size())
                    throw std::runtime_error("DYNAMIC_PREFILL_OUTPUT_DUPLICATE");
                found=j;
            }
        }
        if(found==prefill->outputs.size())
            throw std::runtime_error("DYNAMIC_PREFILL_KV_SOURCE_MISSING");
        const auto& src=prefill->outputs[found];
        const auto& dst=decode->inputs[link.input_index];
        const auto& qs=tensor_quant(src);
        const auto& qd=tensor_quant(dst);
        const auto* dims=tensor_dims(src);
        if(tensor_rank(src)!=4 || tensor_dtype(src)!=QNN_DATATYPE_SFIXED_POINT_8 ||
           dims[0]!=1 || dims[1]!=1 ||
           (link.key?(dims[2]!=link.channels || dims[3]!=128):
                     (dims[2]!=128 || dims[3]!=link.channels)) ||
           tensor_nbytes(src)!=link.channels*128 ||
           qs.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           qd.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
           qs.scaleOffsetEncoding.scale!=qd.scaleOffsetEncoding.scale ||
           qs.scaleOffsetEncoding.offset!=qd.scaleOffsetEncoding.offset ||
           qd.scaleOffsetEncoding.offset!=0)
            throw std::runtime_error("DYNAMIC_PREFILL_KV_QUANT_OR_SHAPE_INVALID");
        auto& cache=decode->input_buffers[link.input_index];
        if(cache.size()!=tensor_nbytes(dst) ||
           !std::all_of(cache.begin(),cache.end(),
                        [](uint8_t b){return b==0;}))
            throw std::runtime_error("DYNAMIC_DECODE_CACHE_NOT_INITIAL_ZERO");
        gemma4::update_kv_block(cache,prefill->output_buffers[found],
            link.channels,4096,0,128,seed_pos,
            link.key?gemma4::KvLayout::Key:gemma4::KvLayout::Value);
        ++prefill_sources;
        cache_bytes+=cache.size();
    }
    if(prefill_sources!=30 || kv_indexes.size()!=30)
        throw std::runtime_error("DYNAMIC_KV_LINKS_INCOMPLETE");
    std::cout<<"DYNAMIC_KV_BRIDGE_OK: from_prefill=30 prefix_rows="<<seed_pos
             <<" cache_bytes="<<cache_bytes
             <<" no_saved_decode_KV=1\n"<<std::flush;
    const auto bench8_kv_bridge_done=std::chrono::steady_clock::now();
    if(options.dynamic_benchmark) {
        const auto ms=[](auto end,auto start)->double {
            return std::chrono::duration<double,std::milli>(end-start).count();
        };
        std::cout<<"BENCH8_PREFILL: real_tokens="<<seed_pos
                 <<" graph_tokens=128"
                 <<" cpu_inputs_ms="<<ms(bench8_cpu_prefill_done,bench8_cpu_prefill_start)
                 <<" runtime_contexts_ms="<<ms(bench8_context_done,bench8_context_start)
                 <<" npu_execute_ms="<<ms(bench8_prefill_npu_done,bench8_prefill_npu_start)
                 <<" kv_bridge_ms="<<ms(bench8_kv_bridge_done,bench8_kv_bridge_start)
                 <<" warmed_up=0\n"<<std::flush;
    }

    // Decode now takes only fresh CPU non-KV inputs and linked RAM KV.
    auto nonkv=[&](const std::string& name,Qnn_DataType_t dtype,
                   size_t bytes,std::vector<uint8_t> value) {
        size_t found=decode->inputs.size();
        for(size_t i=0;i<decode->inputs.size();++i) {
            const char* actual=tensor_name(decode->inputs[i]);
            if(actual && name==actual) {
                if(found!=decode->inputs.size())
                    throw std::runtime_error("DYNAMIC_NONKV_DUPLICATE: "+name);
                found=i;
            }
        }
        if(found==decode->inputs.size() || kv_indexes.count(found) ||
           tensor_dtype(decode->inputs[found])!=dtype ||
           tensor_nbytes(decode->inputs[found])!=bytes ||
           value.size()!=bytes)
            throw std::runtime_error("DYNAMIC_NONKV_METADATA_MISMATCH: "+name);
        decode->input_buffers[found]=std::move(value);
        return found;
    };
    std::unique_ptr<Stage82KvMem> stage82_mem;
    if(options.dynamic_memhandle_kv)
        stage82_mem=std::make_unique<Stage82KvMem>(runtime,*decode,links);
    // Stage 8.1: fixed-length SYNTHETIC load test. Each QNN graph is executed
    // for the same real inputs as Stage 7.x, but after the header LF107,
    // force token ID 236778 ("2") on EVERY subsequent Decode position.
    // This isolates graph/KV costs with 256 steps independent of EOS/ties.
    // IMPORTANT: This branch is opt-in and never renders generated text.
    if(options.dynamic_bench_fixed_steps_set) {
        constexpr int32_t fixed_token=236778;
        const uint32_t steps=options.dynamic_bench_fixed_steps;
        if(seed_pos+steps>4096)
            throw std::runtime_error("BENCH81_KV_CAPACITY_EXCEEDED");
        const auto workload_start=std::chrono::steady_clock::now();
        double cpu_total=0,decode_total=0,head_call_total=0;
        double feedback_total=0,steps_total=0,first_step_ms=0;
        const auto ms=[](auto end,auto begin)->double {
            return std::chrono::duration<double,std::milli>(end-begin).count();
        };
        std::cout<<"BENCH81_MODE: requested_steps="<<steps
                 <<" prefill_tokens="<<seed_pos
                 <<" decode_first_token="<<seed
                 <<" decode_replay_token="<<fixed_token
                 <<" context_capacity=4096"
                 <<" kv_binding="<<(stage82_mem?"memhandle":"raw")
                 <<" synthetic_fixed_input=1 no_eos_stop=1"
                 <<" no_argmax_tie_stop=1 kv_feedback_each_step=1"
                 <<" excludes_context_load_from_decode_rate=1\n"<<std::flush;
        for(uint32_t step=1;step<=steps;++step) {
            const size_t pos=seed_pos+(step-1);
            const int32_t input_token=(step==1)?seed:fixed_token;
            const auto t0=std::chrono::steady_clock::now();
            std::set<size_t> nonkv_indexes;
            nonkv_indexes.insert(nonkv(
                "decode_embeddings_0",QNN_DATATYPE_FLOAT_32,6144,
                gemma4::compute_packed_embedding(
                    options.dynamic_section0,0,input_token)));
            nonkv_indexes.insert(nonkv(
                "decode_per_layer_embeddings_0",QNN_DATATYPE_FLOAT_32,35840,
                gemma4::compute_packed_embedding(
                    options.dynamic_section2,2,input_token)));
            const auto aux=gemma4::run_decode_aux(
                options.dynamic_section1,static_cast<int32_t>(pos),input_token);
            if(aux.size()!=6)
                throw std::runtime_error("BENCH81_DECODE_AUX_NOT_SIX");
            const std::array<size_t,6> aux_bytes{512,512,256,256,4097,4097};
            for(size_t i=0;i<aux.size();++i)
                nonkv_indexes.insert(nonkv(
                    aux[i].qnn_name,QNN_DATATYPE_SFIXED_POINT_8,
                    aux_bytes[i],std::vector<uint8_t>(aux[i].bytes)));
            if(nonkv_indexes.size()!=8 ||
               nonkv_indexes.size()+kv_indexes.size()!=38)
                throw std::runtime_error("BENCH81_DECODE_INPUTS_INCOMPLETE");
            for(size_t i=0;i<decode->inputs.size();++i)
                bind_raw(decode->inputs[i],decode->input_buffers[i]);
            if(stage82_mem)stage82_mem->bind_all();
            const auto t1=std::chrono::steady_clock::now();
            const bool profile_sample=options.dynamic_qnn_profile_set &&
                (step==1 || step==2 || step==16 || step==64 ||
                 step==128 || step==steps);
            if(profile_sample)
                runtime.profile_next("decode",step,options.dynamic_qnn_profile_detailed);
            runtime.execute(decode->handle,decode->inputs,decode->outputs);
            const auto t2=std::chrono::steady_clock::now();
            if(profile_sample)
                runtime.profile_next("head",step,options.dynamic_qnn_profile_detailed);
            execute_chained_decode_head(runtime,*decode,*head,hidden,true);
            const auto t3=std::chrono::steady_clock::now();
            size_t feedback_bytes=0,cache_updates=0;
            for(const auto& link:links) {
                const auto& slice=decode->output_buffers[link.output_index];
                auto& cache=decode->input_buffers[link.input_index];
                gemma4::update_kv_slot(
                    cache,slice,link.channels,4096,pos,
                    link.key?gemma4::KvLayout::Key:
                             gemma4::KvLayout::Value);
                if(stage82_mem)stage82_mem->mirror_slot(link,slice,pos);
                feedback_bytes+=slice.size();
                ++cache_updates;
            }
            if(feedback_bytes!=9216 || cache_updates!=30)
                throw std::runtime_error("BENCH81_KV_FEEDBACK_INVALID");
            const auto t4=std::chrono::steady_clock::now();
            if(stage82_mem && step==steps)
                stage82_mem->verify_equal("final");
            if(options.dynamic_bench_checksums) {
                const auto fnv=[](uint64_t& hash,
                                  const std::vector<uint8_t>& v) {
                    for(const uint8_t byte:v) {
                        hash^=byte;
                        hash*=uint64_t{1099511628211};
                    }
                };
                uint64_t decode_hash=uint64_t{14695981039346656037ULL};
                uint64_t head_hash=uint64_t{14695981039346656037ULL};
                for(const auto& v:decode->output_buffers)
                    fnv(decode_hash,v);
                fnv(head_hash,head->output_buffers.at(0));
                std::cout<<"STAGE82_CHECKSUM: step="<<step
                         <<" decode_fnv="<<decode_hash
                         <<" head_fnv="<<head_hash
                         <<" comparable_raw_memhandle=1\n"<<std::flush;
            }
            const double cpu=ms(t1,t0),dec=ms(t2,t1);
            const double head_call=ms(t3,t2),feedback=ms(t4,t3);
            const double total=ms(t4,t0);
            cpu_total+=cpu;
            decode_total+=dec;
            head_call_total+=head_call;
            feedback_total+=feedback;
            steps_total+=total;
            if(step==1)first_step_ms=total;
            std::cout<<"BENCH81_STEP: step="<<step
                     <<" pos="<<pos<<" input_token="<<input_token
                     <<" cpu_nonkv_ms="<<cpu
                     <<" npu_decode_ms="<<dec
                     <<" head_call_ms="<<head_call
                     <<" kv_feedback_ms="<<feedback
                     <<" step_wall_ms="<<total
                     <<" kv_updated=1 kv_tensors=30"
                     <<" kv_slice_bytes="<<feedback_bytes
                     <<" no_sample_logits_used=1\n"<<std::flush;
        }
        const double workload_wall_ms=ms(
            std::chrono::steady_clock::now(),workload_start);
        std::cout<<"BENCH81_OK: executed_steps="<<steps
                 <<" kv_updates="<<steps
                 <<" kv_tensors_per_step=30"
                 <<" last_written_pos="<<(seed_pos+steps-1)
                 <<" fixed_token="<<fixed_token
                 <<" prefill_tokens="<<seed_pos
                 <<" cpu_nonkv_total_ms="<<cpu_total
                 <<" npu_decode_total_ms="<<decode_total
                 <<" head_call_total_ms="<<head_call_total
                 <<" kv_feedback_total_ms="<<feedback_total
                 <<" measured_steps_total_ms="<<steps_total
                 <<" first_step_ms="<<first_step_ms
                 <<" decode_steps_per_s="<<(steps_total>0?1000.0*steps/steps_total:0.0)
                 <<" workload_wall_ms="<<workload_wall_ms
                 <<" synthetic_not_text_generation=1"
                 <<" kv_binding="<<(stage82_mem?"memhandle":"raw")
                 <<" qnn_profile="<<(options.dynamic_qnn_profile_set?
                    (options.dynamic_qnn_profile_detailed?"detailed":"basic"):"off")
                 <<" full_KV_verified="<<(stage82_mem?1:0)
                 <<"\n"<<std::flush;
        return;
    }

    size_t executed=0,produced=0,tie_resolutions=0;
    int32_t token=seed;
    size_t position=seed_pos;
    bool eos=false;
    for(uint32_t step=1;step<=options.dynamic_max_tokens;++step) {
        const auto bench8_step_start=std::chrono::steady_clock::now();
        std::set<size_t> nonkv_indexes;
        nonkv_indexes.insert(nonkv(
            "decode_embeddings_0",QNN_DATATYPE_FLOAT_32,6144,
            gemma4::compute_packed_embedding(options.dynamic_section0,0,token)));
        nonkv_indexes.insert(nonkv(
            "decode_per_layer_embeddings_0",QNN_DATATYPE_FLOAT_32,35840,
            gemma4::compute_packed_embedding(options.dynamic_section2,2,token)));
        const auto next_aux=gemma4::run_decode_aux(
            options.dynamic_section1,static_cast<int32_t>(position),token);
        if(next_aux.size()!=6)
            throw std::runtime_error("DYNAMIC_DECODE_AUX_NOT_SIX");
        const std::array<size_t,6> aux_bytes{512,512,256,256,4097,4097};
        for(size_t i=0;i<next_aux.size();++i)
            nonkv_indexes.insert(nonkv(next_aux[i].qnn_name,
                QNN_DATATYPE_SFIXED_POINT_8,aux_bytes[i],
                std::vector<uint8_t>(next_aux[i].bytes)));
        if(nonkv_indexes.size()!=8 || nonkv_indexes.size()+kv_indexes.size()!=38)
            throw std::runtime_error("DYNAMIC_DECODE_INPUTS_INCOMPLETE");
        for(size_t i=0;i<decode->inputs.size();++i)
            bind_raw(decode->inputs[i],decode->input_buffers[i]);
        const auto bench8_cpu_step_done=std::chrono::steady_clock::now();
        runtime.execute(decode->handle,decode->inputs,decode->outputs);
        const auto bench8_decode_done=std::chrono::steady_clock::now();
        execute_chained_decode_head(runtime,*decode,*head,hidden,true);
        const auto bench8_head_done=std::chrono::steady_clock::now();
        executed=step;
        const auto& logits=head->output_buffers.at(0);
        if(logits.size()!=262144 ||
           tensor_dtype(head->outputs.at(0))!=QNN_DATATYPE_SFIXED_POINT_8)
            throw std::runtime_error("DYNAMIC_LOGITS_INVALID");
        int best=-129;int32_t id=-1;size_t ties=0;
        for(size_t i=0;i<logits.size();++i) {
            const int q=logits[i]<128?int(logits[i]):int(logits[i])-256;
            if(q>best){best=q;id=static_cast<int32_t>(i);ties=1;}
            else if(q==best)++ties;
        }
        if(ties>1) {
            // Print the tied IDs from the actual current LM Head output,
            // never imply that INT8 q distinguishes their FP32 ordering.
            constexpr size_t kMaxDisplayed=64;
            std::ostringstream candidates;
            size_t emitted=0;
            for(size_t i=0;i<logits.size();++i) {
                const int q=logits[i]<128?int(logits[i]):int(logits[i])-256;
                if(q!=best)continue;
                if(emitted<kMaxDisplayed) {
                    if(emitted)candidates<<",";
                    candidates<<i;
                }
                ++emitted;
            }
            if(emitted!=ties)
                throw std::runtime_error("DYNAMIC_TIE_COUNT_CHANGED");
            std::cout<<"DYNAMIC_TIE_CANDIDATES: step="<<step
                     <<" max_q="<<best<<" ties="<<ties
                     <<" candidate_ids="<<candidates.str()
                     <<" shown="<<std::min(emitted,kMaxDisplayed)
                     <<" truncated="<<(emitted>kMaxDisplayed?1:0)
                     <<" quantized_logits_only=1"
                     <<" no_fp32_ranking_claim=1\n"<<std::flush;
        }
        std::cout<<"DYNAMIC_GENERATED_TOKEN: step="<<step
                 <<" response_pos="<<(position+1)
                 <<" id="<<id<<" max_q="<<best
                 <<" ties="<<ties<<" input_pos="<<position
                 <<" input_token="<<token<<"\n"<<std::flush;
        const auto bench8_sample_done=std::chrono::steady_clock::now();
        const auto bench8_report_step=[&](const char* reason,bool with_feedback) {
            if(!options.dynamic_benchmark)return;
            const auto finish=std::chrono::steady_clock::now();
            const auto ms=[](auto end,auto begin)->double {
                return std::chrono::duration<double,std::milli>(end-begin).count();
            };
            std::cout<<"BENCH8_STEP: step="<<step
                     <<" cpu_nonkv_ms="<<ms(bench8_cpu_step_done,bench8_step_start)
                     <<" npu_decode_ms="<<ms(bench8_decode_done,bench8_cpu_step_done)
                     <<" head_call_ms="<<ms(bench8_head_done,bench8_decode_done)
                     <<" sampling_and_logging_ms="<<ms(bench8_sample_done,bench8_head_done)
                     <<" kv_feedback_ms="<<(with_feedback?ms(finish,bench8_sample_done):0.0)
                     <<" step_wall_ms="<<ms(finish,bench8_step_start)
                     <<" feedback_written="<<(with_feedback?1:0)
                     <<" stop="<<reason<<"\n"<<std::flush;
        };
        if(options.dynamic_benchmark && step==1) {
            const double to_first=std::chrono::duration<double,std::milli>(
                bench8_sample_done-bench8_job_started).count();
            std::cout<<"BENCH8_FIRST_TOKEN: from_job_start_ms="<<to_first
                     <<" includes_metadata_cpu_and_load=1"
                     <<" excludes_external_BPE_and_process_start=1\n"<<std::flush;
        }
        if(ties!=1) {
            if(!options.dynamic_allow_tie_lowest) {
                std::cout<<"DYNAMIC_STOP_TIE: step="<<step
                         <<" max_q="<<best<<" ties="<<ties
                         <<" no_arbitrary_tiebreak=1\n"<<std::flush;
                bench8_report_step("tie",false);
                return;
            }
            ++tie_resolutions;
            std::cout<<"DYNAMIC_TIE_RESOLVED: step="<<step
                     <<" ties="<<ties<<" chosen_id="<<id
                     <<" policy=lowest-id-experimental"
                     <<" no_fp32_ranking_claim=1\n"<<std::flush;
        }
        if(id==1 || id==106 || id==50) {
            eos=true;
            std::cout<<"DYNAMIC_STOP_EOS: step="<<step
                     <<" eos_id="<<id
                     <<" eos_not_submitted_to_decode=1\n"<<std::flush;
            bench8_report_step("eos",false);
            break;
        }
        ++produced;
        if(step==options.dynamic_max_tokens) {
            bench8_report_step("limit",false);
            break;
        }
        if(position+1>=4096)
            throw std::runtime_error("DYNAMIC_KV_CAPACITY_EXCEEDED");
        size_t feedback_bytes=0;
        for(const auto& link:links) {
            const auto& slice=decode->output_buffers[link.output_index];
            auto& cache=decode->input_buffers[link.input_index];
            gemma4::update_kv_slot(cache,slice,link.channels,4096,position,
                link.key?gemma4::KvLayout::Key:gemma4::KvLayout::Value);
            feedback_bytes+=slice.size();
        }
        if(feedback_bytes!=9216)
            throw std::runtime_error("DYNAMIC_KV_FEEDBACK_BAD_BYTES");
        token=id;
        ++position;
        bench8_report_step("continue",true);
    }
    std::cout<<"DYNAMIC_GENERATION_OK: npu_decode_steps="<<executed
             <<" text_tokens="<<produced
             <<" tie_resolutions="<<tie_resolutions
             <<" stop_reason="<<(eos?"eos":"limit")
             <<" saved_QNN_inputs_used=0 saved_references_used=0"
             <<" all_CPU_nonKV=1 prefill_KV_in_RAM=1\n"<<std::flush;
}
