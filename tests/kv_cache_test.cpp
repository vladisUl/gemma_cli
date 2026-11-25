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
    std::cout << "KV_CACHE_UNIT_TEST_OK: K/V layouts, bounds, inferred slot\n";
    return 0;
}
