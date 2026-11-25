#include "kv_cache.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char* msg) {
    if (!condition) throw std::runtime_error(msg);
}
int main() {
    using gemma4::KvLayout;
    using gemma4::update_kv_slot;
    std::vector<uint8_t> k{10,11,12,13,20,21,22,23,30,31,32,33};
    update_kv_slot(k, {99,88,77}, 3,4,2,KvLayout::Key);
    require(k == std::vector<uint8_t>({10,11,99,13,20,21,88,23,30,31,77,33}),
            "Key must overwrite only strided column 2");
    std::vector<uint8_t> v{10,11,12,20,21,22,30,31,32,40,41,42};
    update_kv_slot(v, {99,88,77}, 3,4,2,KvLayout::Value);
    require(v == std::vector<uint8_t>({10,11,12,20,21,22,99,88,77,40,41,42}),
            "Value must overwrite only contiguous row 2");
    update_kv_slot(k, {1,2,3}, 3,4,0,KvLayout::Key);
    require(k[0]==1 && k[4]==2 && k[8]==3,"Key pos 0");
    update_kv_slot(v, {1,2,3}, 3,4,3,KvLayout::Value);
    require(v[9]==1 && v[10]==2 && v[11]==3,"Value pos 3");
    bool thrown = false;
    try { update_kv_slot(k, {1,2,3},3,4,4,KvLayout::Key); }
    catch(const std::invalid_argument&) { thrown=true; }
    require(thrown,"Out-of-range position must throw");
    thrown=false;
    try { update_kv_slot(v,{1,2},3,4,2,KvLayout::Value); }
    catch(const std::invalid_argument&) { thrown=true; }
    require(thrown,"Wrong slice length must throw");
    // Slot inference uses both cache layouts and must reject a two-step diff.
    const std::vector<uint8_t> k0{10,11,12,13,20,21,22,23,30,31,32,33};
    auto k1 = k0;
    update_kv_slot(k1,{44,55,66},3,4,2,KvLayout::Key);
    const auto kp = gemma4::find_changed_kv_slot(k0,k1,3,4,KvLayout::Key);
    require(kp && *kp == 2,"K slot inference");
    require(!gemma4::find_changed_kv_slot(k0,k0,3,4,KvLayout::Key),
            "Identical K must not produce a guessed slot");
    const std::vector<uint8_t> v0{10,11,12,20,21,22,30,31,32,40,41,42};
    auto v1 = v0;
    update_kv_slot(v1,{44,55,66},3,4,2,KvLayout::Value);
    const auto vp = gemma4::find_changed_kv_slot(v0,v1,3,4,KvLayout::Value);
    require(vp && *vp == 2,"V slot inference");
    k1[0] ^= 1;
    thrown=false;
    try { (void)gemma4::find_changed_kv_slot(k0,k1,3,4,KvLayout::Key); }
    catch(const std::invalid_argument&) {thrown=true;}
    require(thrown,"K must reject differences at two distinct slots");
    v1[11] ^= 1;
    thrown=false;
    try { (void)gemma4::find_changed_kv_slot(v0,v1,3,4,KvLayout::Value); }
    catch(const std::invalid_argument&) {thrown=true;}
    require(thrown,"V must reject differences at two distinct slots");
    // 6-slot / 3-channel cache, with 4 exported Prefill rows.
    // Only 2 non-padding positions are copied at global positions 1 and 2.
    const std::vector<uint8_t> prefill_k{
        10,11,12,13, 20,21,22,23, 30,31,32,33
    }; // channel-major 3x4
    std::vector<uint8_t> block_k(3*6, 99);
    gemma4::update_kv_block(block_k,prefill_k,3,6,1,4,2,KvLayout::Key);
    require(block_k==std::vector<uint8_t>({
        99,10,11,99,99,99,
        99,20,21,99,99,99,
        99,30,31,99,99,99}),"Prefill K block channel-major placement");

    const std::vector<uint8_t> prefill_v{
        10,20,30, 11,21,31, 12,22,32, 13,23,33
    }; // row-major 4x3
    std::vector<uint8_t> block_v(3*6, 99);
    gemma4::update_kv_block(block_v,prefill_v,3,6,1,4,2,KvLayout::Value);
    require(block_v==std::vector<uint8_t>({
        99,99,99, 10,20,30, 11,21,31,
        99,99,99, 99,99,99, 99,99,99}),"Prefill V block contiguous placement");

    // Distinguish a Prefill block from repeated Decode slots.
    std::vector<uint8_t> assembled_k(3*6,0);
    std::vector<uint8_t> assembled_v(3*6,0);
    gemma4::update_kv_block(assembled_k,prefill_k,3,6,0,4,2,KvLayout::Key);
    gemma4::update_kv_block(assembled_v,prefill_v,3,6,0,4,2,KvLayout::Value);
    for(size_t r=0;r<2;++r) {
        std::vector<uint8_t> ks(3),vs(3);
        for(size_t c=0;c<3;++c) {
            ks[c]=prefill_k[c*4+r];
            vs[c]=prefill_v[r*3+c];
        }
        std::vector<uint8_t> one_k(3*6,0),one_v(3*6,0);
        gemma4::update_kv_slot(one_k,ks,3,6,r,KvLayout::Key);
        gemma4::update_kv_slot(one_v,vs,3,6,r,KvLayout::Value);
        for(size_t c=0;c<3;++c) {
            require(one_k[c*6+r]==assembled_k[c*6+r],"K block matches slot primitive");
            require(one_v[r*3+c]==assembled_v[r*3+c],"V block matches slot primitive");
        }
    }
    thrown=false;
    try {gemma4::update_kv_block(block_k,prefill_k,3,6,5,4,2,KvLayout::Key);}
    catch(const std::invalid_argument&){thrown=true;}
    require(thrown,"Block insertion may not pass cache capacity");
    thrown=false;
    try {gemma4::update_kv_block(block_v,prefill_v,3,6,0,4,5,KvLayout::Value);}
    catch(const std::invalid_argument&){thrown=true;}
    require(thrown,"Block insertion may not include unprovided rows");
    thrown=false;
    try {gemma4::update_kv_block(block_k,{1,2,3},3,6,0,4,2,KvLayout::Key);}
    catch(const std::invalid_argument&){thrown=true;}
    require(thrown,"Wrong Prefill K block length must be rejected");
    // Linked Prefill-Decode proof: the first Decode slice must be written at
    // absolute slot 2, the second at slot 3, preserving the Prefill prefix.
    std::vector<uint8_t> running_k=assembled_k;
    std::vector<uint8_t> running_v=assembled_v;
    gemma4::update_kv_slot(running_k,{71,72,73},3,6,2,KvLayout::Key);
    gemma4::update_kv_slot(running_v,{81,82,83},3,6,2,KvLayout::Value);
    gemma4::update_kv_slot(running_k,{74,75,76},3,6,3,KvLayout::Key);
    gemma4::update_kv_slot(running_v,{84,85,86},3,6,3,KvLayout::Value);
    for(size_t c=0;c<3;++c) {
        require(running_k[c*6]==assembled_k[c*6] &&
                running_k[c*6+1]==assembled_k[c*6+1],
                "Both KV feedback steps preserve Prefill K prefix");
        require(running_k[c*6+2]==71+c &&
                running_k[c*6+3]==74+c,
                "Consecutive K feedback slots 2/3");
        require(running_k[c*6+4]==0 &&
                running_k[c*6+5]==0,
                "No future K slot changed");
    }
    require(running_v==std::vector<uint8_t>({
        10,20,30, 11,21,31, 81,82,83, 84,85,86, 0,0,0, 0,0,0
    }),"Consecutive V feedback slots 2/3 with Prefill prefix retained");
    std::cout << "KV_CACHE_UNIT_TEST_OK: K/V layouts, bounds, inferred slot\n";
    return 0;
}
