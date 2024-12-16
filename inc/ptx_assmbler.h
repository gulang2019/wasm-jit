#pragma once

#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <cassert>
#include "ir.h"
#include <iostream>

typedef size_t reg_t;

constexpr size_t N_ENV = 3;
constexpr const char *ENV_PARAMS[] = {
    "%tid.x",  "%ntid.x", "%ctaid.x" 
};

enum PtxType {
    U32, U64, S32, S64,
    F32, F64, B32, B64
};

inline PtxType wasm_to_ptx_type(const wasm_type_t t) {
    switch (t) {
        case WASM_TYPE_I32: return S32;
        case WASM_TYPE_F64: return F64;
        default: throw std::runtime_error("invalid wasm type to render");
    }
}

inline const char *render_ptype(PtxType t) {
    switch (t) {
        case U32: return "u32";
        case U64: return "u64";
        case S32: return "u64";
        case S64: return "s64";
        case F32: return "f32";
        case F64: return "f64";
        case B32: return "b32";
        case B64: return "b64";
    }
    return "";
}

inline std::string render_reg(wasm_type_t t, reg_t reg = -1) {
    const char* t_str;
    switch (t) {
        case WASM_TYPE_I32: 
            t_str = "%rd";
            break;
        case WASM_TYPE_F64: 
            t_str = "%fd";
            break;
        default:
            throw std::runtime_error("invalid wasm type to render");
    }
    if (reg == -1) return t_str;
    return std::string(t_str) + std::to_string(reg);
}

// static std::string local_name(const reg_t reg) {
//     return "%local" + std::to_string(reg);
// }

struct SValue {
    // reg
    wasm_type_t type;
    reg_t local;

    SValue(wasm_type_t t, reg_t loc): type(t), local(loc) {}
    [[nodiscard]] std::string str() const {
        // if (local < N_ENV) return ENV_PARAMS[local];
        return render_reg(type, local);
    }
    [[nodiscard]] SValue with_reg(reg_t new_reg) const { return {type, new_reg}; }
};

// assert no nondeterministic values in stack, e.g., no cfg block labels
// change this in the future
class PtxStack {
    

public:
    PtxStack() {reset();}
    [[nodiscard]] const SValue& at(const size_t loc) const { return values.at(loc); }
    void reset() { 
        values.clear(); 
        ref_counts.clear();
    }
    SValue& push(wasm_type_t t) {
        // this is the only place where we allocate a new register
        if (ref_counts.find(t) == ref_counts.end()) {
            ref_counts[t] = std::vector<int>();
        }
        for (reg_t i = 0; i < ref_counts[t].size(); i++) {
            if (ref_counts[t][i] == 0) {
                ref_counts[t][i] = 1;
                values.push_back({t, i});
                return values.back();
            }
        }
        ref_counts[t].push_back(1);
        values.push_back({t, ref_counts[t].size() - 1});
        return values.back();
    }
    const SValue& push(const SValue &v) {
        // only local.get calls here.
        values.push_back(v); 
        ref_counts[v.type][v.local] ++;
        return v;
    }
    SValue pop() {
        const auto v = values.back();
        values.pop_back();
        // ref_counts[v.type][v.local] --;
        return v;
    }

    const SValue& set(const size_t loc, const SValue &v) {
        // only local.set calls here.
        auto& old = values[loc];
        // ref_counts[old.type][old.local] --;
        old = v;
        ref_counts[v.type][v.local] ++;
        return old;
    }

    void emit_reg_alloc(std::ostream &ss) {
        // we reserve a register for the environment
        ss << ".reg .i32 %r0;\n"; 
        for (auto &[t, refs]: ref_counts) {
            ss << ".reg ." << render_ptype(wasm_to_ptx_type(t)) << " ";
            ss << render_reg(t) << "<" << refs.size() << ">;\n";
        }
    }

private:
    std::vector<SValue> values;
    std::unordered_map<wasm_type_t, std::vector<int> > ref_counts;
};

class PtxAsm {

public:
    PtxAsm() = default;
    void reset() { ss.str(""); ss.clear(); }
    
    std::string build(PtxStack& stack) { 
        return ss_prologue.str() + ss.str(); }
    
    void gen_headers() {
        ss_prologue << ".version 7.8\n";
        ss_prologue << ".target sm_80\n";
        ss_prologue << ".address_size 64\n";
        ss_prologue << "\n";
    }

    void gen_func_start(const char* name, 
                        const FuncDecl &f, 
                        PtxStack& stack) {
        char _name[256];
        strcpy(_name, name);
        int idx;
        for (idx = strlen(_name) -1; _name[idx] != '-' && idx >= 0; idx--);
        if (idx < 0) {
            std::stringstream ss;
            ss << "Invalid function name: " << name << ", expected: FUNC_NAME-[p|i]," 
            << " ([p|i] indicates if the parameter is a pointer)" << std::endl;
            throw(std::runtime_error(ss.str()));
            exit(1);
        }
        _name[idx] = '\0';
        const char* arg_tags = &_name[idx+1];

        ss_prologue << ".visible .entry " << _name << "(\n";

        int n_args = strlen(arg_tags);
        assert((n_args + N_ENV) == f.sig->params.size());
        int pid = 0;
        ss << "// Prologue Begins;\n";
        for (auto t: f.sig->params) {
            // environment parameters
            if (pid < N_ENV) {
                auto& v = stack.push(WASM_TYPE_I32);
                ss << "mov.u32 %r0, " << ENV_PARAMS[pid] << ";\n";
                ss << "cvt.u64.u32 " << v.str() << ", %r0;\n";
                pid ++;
                continue;
            }

            ss_prologue << "  .param ." << render_ptype(wasm_to_ptx_type(t)) << " ";
            ss_prologue << _name << "_param_" << pid - N_ENV;
            auto& v = stack.push(t);
            
            ss << "ld.param." << render_ptype(wasm_to_ptx_type(t)) << " ";
            ss << v.str() << ", [" << _name << "_param_" << pid - N_ENV << "];\n";
            if (arg_tags[pid - N_ENV] == 'p'){
                ss << "cvta.to.global.u64 " << v.str() << ", " << v.str() << ";\n";
            } 

            pid++;
            if (pid == f.sig->params.size()) ss_prologue << "\n";
            else ss_prologue << ",\n";
        }
        ss_prologue << ") {\n";
        ss << "// Prologue ends;\n";
    }
    // reg_t gen_local() {
    //     const auto id = n_locals;
    //     ss << ".reg .u32 " << local_name(n_locals++) << "\n";
    //     return id;
    // }
    void gen_finish(PtxStack& stack) { ss << "}\n"; stack.emit_reg_alloc(ss_prologue); }

    // reg_t new_reg(const SValue &v) {
    //     const reg_t r = gen_local();
    //     emit_mov(r, v);
    //     return r;
    // }
    // SValue copy_to_new_reg(const SValue &v) { return v.with_reg(new_reg(v)); }

    void emit_unknown(Opcode_t c) { ss << "<UNKNOWN: [0x" << std::hex << c << "]>\n"; }

    // arithmetics
    // void emit_mov(reg_t dest, const SValue &from) {
    //     ss << "mov." << render_ptype(wasm_to_ptx_type(from.type)) << " ";
    //     ss << local_name(dest) << ", " << from.str() << "\n";
    // }
    void emit_mov_i32(SValue& r, int32_t v) {
        ss << "mov.s64 " << r.str() << ", " << v << "\n";
    }

    void emit_binop(const char *op, const SValue& dest, const SValue &src_a, const SValue &src_b) {
        ss << op << "." << render_ptype(wasm_to_ptx_type(src_a.type)) << " ";
        ss << dest.str() << ", " << src_a.str() << ", " << src_b.str() << "\n";
    }

    void emit_load(const char *mode, const SValue& r, const SValue &v) {
        ss << "ld." << mode << "." << render_ptype(wasm_to_ptx_type(v.type));
        ss << " " << r.str() << ", [";
        ss << v.str() << "]\n";
    }

    void emit_store(const char *mode, const SValue &addr, const SValue &v) {
        ss << "store." << mode << "." << render_ptype(wasm_to_ptx_type(v.type));
        ss << " [" << addr.str() << "], ";
        ss << v.str() << "\n";
    }

    void emit_comment(const std::string &comment) { 
        ss << "// " << comment << "\n"; 
    }

private:
    std::stringstream ss;
    std::stringstream ss_prologue;
};
