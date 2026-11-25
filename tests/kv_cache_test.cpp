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
    std::cout << "KV_CACHE_UNIT_TEST_OK: K/V layout and bounds\n";
    return 0;
}
