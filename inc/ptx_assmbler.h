#pragma once

#include <vector>
#include <sstream>

#include "ir.h"

typedef size_t reg_t;

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
        case S32: return "s32";
        case S64: return "s64";
        case F32: return "f32";
        case F64: return "f64";
        case B32: return "b32";
        case B64: return "b64";
    }
    return "";
}

static std::string local_name(const reg_t reg) {
    return "%local" + std::to_string(reg);
}

struct SValue {
    bool is_imm;

    // reg
    wasm_type_t type;
    reg_t local = 0;

    // imm
    int32_t v_i = 0;
    double v_f = 0;

    SValue(reg_t loc, wasm_type_t type): is_imm(false), local(loc), type(type) {}
    explicit SValue(int32_t v): is_imm(true), type(WASM_TYPE_I32), v_i(v) {}
    explicit SValue(double v): is_imm(true), type(WASM_TYPE_F64), v_f(v) {}
    [[nodiscard]] std::string str() const {
        if (is_imm) {
            if (type == WASM_TYPE_I32) return std::to_string(v_i);
            if (type == WASM_TYPE_F64) return std::to_string(v_f);
            return "<INVALID VALUE>";
        }
        return local_name(local);
    }
};

// assert no nondeterministic values in stack, e.g., no cfg block labels
// change this in the future
class PtxStack {

public:
    [[nodiscard]] SValue at(const size_t loc) const { return values[loc]; }
    void reset() { values.clear(); }
    void push(const SValue &v) { values.push_back(v); }
    SValue pop() {
        const auto v = values.back();
        values.pop_back();
        return v;
    }

private:
    std::vector<SValue> values;
};

class PtxAsm {

public:
    PtxAsm() = default;
    void reset() { ss.str(""); ss.clear(); }
    std::string build() const { return ss.str(); }
    void gen_headers() {
        ss << ".version 8.4\n";
        ss << ".target sm_89\n";
        ss << ".address_size 64\n";
        ss << "\n";
    }
    void gen_func_start(const std::string &name, const FuncDecl &f) {
        ss << ".visible .entry " << name << "(\n";
        size_t i = 0;
        for (auto p: f.sig->params) {
            ss << "  .param ." << render_ptype(wasm_to_ptx_type(p)) << " ";
            ss << local_name(i);
            i++;

            if (i == f.sig->params.size()) ss << "\n";
            else ss << ",\n";
        }
        ss << ") {\n";
    }
    void gen_local(size_t i) { ss << ".reg .u32 %" << local_name(i) << "\n"; }
    void gen_func_finish() { ss << "}\n"; }

    reg_t gen_scratch_reg(reg_t &locals) {
        ss << ".reg .u32 %" << local_name(locals) << "\n";
        return locals++;
    }

    reg_t reg(const SValue &v, const reg_t _default) {
        if (v.is_imm) {

            ss << "mov." << render_ptype(wasm_to_ptx_type(v.type)) << " ";
            ss << local_name(_default) << ", " << v.str();

            return _default;
        } else return v.local;
    }

    // arithmetics
    void emit_add(PtxType t, uint32_t dest, SValue a, SValue b) {
        ss << "add." << render_ptype(t) << " ";
        ss << "%" << dest << ", " << a.str() << ", " << b.str();
    }

private:
    std::stringstream ss;
};
