#pragma once

#include <vector>
#include <sstream>

#include "ir.h"

typedef size_t reg_t;

constexpr size_t N_ENV = 3;
constexpr const char *ENV_PARAMS[] = {
    "%ctaid.x", "%ntid.x", "%tid.x"
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
    // reg
    wasm_type_t type;
    reg_t local = 0;

    SValue(reg_t loc, wasm_type_t t): local(loc), type(t) {}
    [[nodiscard]] std::string str() const {
        if (local < N_ENV) return ENV_PARAMS[local];
        return local_name(local);
    }
    [[nodiscard]] SValue with_reg(reg_t new_reg) const { return {new_reg, type}; }
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
    void reset() { n_locals = 0; ss.str(""); ss.clear(); }
    std::string build() const { return ss.str(); }
    void gen_headers() {
        ss << ".version 8.4\n";
        ss << ".target sm_89\n";
        ss << ".address_size 64\n";
        ss << "\n";
    }
    void gen_func_start(const std::string &name, const FuncDecl &f) {
        ss << ".visible .entry " << name << "(\n";
        for (auto p: f.sig->params) {
            auto ref = n_locals++;
            if (ref < N_ENV) continue;
            ss << "  .param ." << render_ptype(wasm_to_ptx_type(p)) << " ";
            ss << local_name(ref);

            if (n_locals == f.sig->params.size()) ss << "\n";
            else ss << ",\n";
        }
        ss << ") {\n";
    }
    reg_t gen_local() {
        const auto id = n_locals;
        ss << ".reg .u32 " << local_name(n_locals++) << "\n";
        return id;
    }
    void gen_func_finish() { ss << "}\n"; }

    reg_t new_reg(const SValue &v) {
        const reg_t r = gen_local();
        emit_mov(r, v);
        return r;
    }
    SValue copy_to_new_reg(const SValue &v) { return v.with_reg(new_reg(v)); }

    void emit_unknown(Opcode_t c) { ss << "<UNKNOWN: [0x" << std::hex << c << "]>\n"; }

    // arithmetics
    void emit_mov(reg_t dest, const SValue &from) {
        ss << "mov." << render_ptype(wasm_to_ptx_type(from.type)) << " ";
        ss << local_name(dest) << ", " << from.str() << "\n";
    }
    void emit_mov_i32(reg_t dest, int32_t v) {
        ss << "mov.s32 " << local_name(dest) << ", " << v << "\n";
    }

    void emit_binop(const char *op, reg_t dest, const SValue &a, const SValue &b) {
        ss << op << "." << render_ptype(wasm_to_ptx_type(a.type)) << " ";
        ss << local_name(dest) << ", " << a.str() << ", " << b.str() << "\n";
    }

    void emit_load(const char *mode, reg_t r, const SValue &v) {
        ss << "ld." << mode << "." << render_ptype(wasm_to_ptx_type(v.type));
        ss << " " << local_name(r) << ", [";
        ss << v.str() << "]\n";
    }

    void emit_store(const char *mode, const SValue &addr, const SValue &v) {
        ss << "ld." << mode << "." << render_ptype(wasm_to_ptx_type(v.type));
        ss << " [" << addr.str() << "], ";
        ss << v.str() << "\n";
    }

private:
    reg_t n_locals = 0;
    std::stringstream ss;
};
