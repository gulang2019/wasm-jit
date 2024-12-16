#include <iostream>
#include <memory>
#include <unordered_set>

#include "ptx_assmbler.h"

void PtxStack::canonicalize(
    PtxAsm& masm,
    std::vector<SValue>*& canonical_values) {
    
    std::unordered_map<std::string, std::pair<std::string, int> > dst_src;
    // printf("Canonicalization #locals %d, #values %d %p\n", n_params, values.size(), canonical_values);
    if (canonical_values == nullptr) {
        canonical_values = new std::vector<SValue>;
        // first, we find out a list of registers that are distinct with each other
        for (size_t i = n_params; i < values.size(); i++) {
            auto& v = values[i];   
            if (dst_src.count(v.str())) { // this register is the destination.
                assert(--ref_counts[v.type][v.local] >= 1);
                auto new_v = _alloc(v.type);
                dst_src[new_v.str()].first = v.str();
                v = new_v;
            }
            else dst_src[v.str()].first = v.str();
            dst_src[v.str()].second ++;
            canonical_values->push_back(v);
        }
    }
    else {
        assert (canonical_values->size() == (values.size() - n_params));
        for (int i = n_params; i < values.size(); i++) {
            auto& v = values[i];
            auto& new_v = canonical_values->at(i - n_params);
            dst_src[new_v.str()].first = v.str();
            ref_counts[v.type][v.local] --;
            dst_src[v.str()].second ++;
            ref_counts[new_v.type][new_v.local] ++;
            v = new_v;
        }
    }
    
    // std::cout << "Canonicalization: " << std::endl;
    // for (auto& [dst, src]: dst_src) {
    //     std::cout << dst << " <- " << src.first << std::endl;
    // }

    // we perform a cyclic shuffling to canonicalize the register names
    std::unordered_set<std::string> cycles;
    std::unordered_set<std::string> visited_set;
    for (auto& [dst, src]: dst_src) {
        if (src.second == 0) { // the root of a chain 
            assert(dst != src.first);
            std::vector<std::string> visited;
            auto cur = dst;
            while (dst_src.count(cur) and !visited_set.count(cur)) {
                visited_set.insert(cur);
                visited.push_back(cur);
                cur = dst_src[cur].first;
            }
            // cur is in a cycle
            SValue v;
            
            for (size_t i = 1; i < visited.size(); i++) {
                auto node = visited[i];
                auto prev_node = visited[i - 1];
                masm.emit_mov(SValue(prev_node), SValue(node));
                if (node == cur) break;
            }
            cycles.insert(cur);
        }
    }
    // std::cout << "Cycles: " << std::endl;
    // for (auto& c: cycles) {
    //     std::cout << c << std::endl;
    // }
    for (auto& [dst, src]: dst_src) {
        if ((cycles.count(dst) or 
            (not visited_set.count(dst))) 
            and src.first != dst) {
            auto v = SValue(dst);
            auto tmp_reg = tmp_regs[v.type];
            auto prev_reg = tmp_reg;
            auto cur_dst = dst;
            while (true) {
                visited_set.insert(cur_dst);
                masm.emit_mov(prev_reg, SValue(cur_dst));
                auto cur_src = dst_src[cur_dst].first;
                if (cur_src == dst) {
                    masm.emit_mov(cur_dst, tmp_reg);
                    break;
                }
                prev_reg = SValue(cur_dst);
                cur_dst = cur_src;
            }
        }
    }
}