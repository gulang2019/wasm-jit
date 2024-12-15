#pragma once

#include <vector>
#include <sstream>

#include "ir.h"

enum PtxType {
    U32, U64, S32, S64,
    F32, F64, B32, B64
};

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

struct SValue {
    bool is_imm;

    // reg
    wasm_type_t type;
    uint32_t local = 0;

    // imm
    int32_t v_i = 0;
    double v_f = 0;

    SValue(int loc, wasm_type_t type): is_imm(false), local(loc), type(type) {}
    explicit SValue(int32_t v): is_imm(true), type(WASM_TYPE_I32), v_i(v) {}
    explicit SValue(double v): is_imm(true), type(WASM_TYPE_F64), v_f(v) {}
    [[nodiscard]] std::string str() const {
        if (is_imm) {
            if (type == WASM_TYPE_I32) return std::to_string(v_i);
            if (type == WASM_TYPE_F64) return std::to_string(v_f);
            return "<INVALID VALUE>";
        }
        return "local" + std::to_string(local);
    }
};

// assert no nondeterministic values in stack, e.g., no cfg block labels
// change this in the future
class PtxStack {

public:
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

    }

    // arithmetics
    void emit_add(PtxType t, uint32_t dest, SValue a, SValue b) {
        ss << "add." << render_ptype(t) << " ";
        ss << "%" << dest << ", " << a.str() << ", " << b.str();
    }

private:
    std::stringstream ss;
};
