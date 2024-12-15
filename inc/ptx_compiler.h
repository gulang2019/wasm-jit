#pragma once

#include "ptx_assmbler.h"
#include "SinglePassCompiler.h"
#include "ir.h"

class PTXCompiler {

public:
    void compile(const FuncDecl &f);

private:
    PtxAsm masm;
    PtxStack stack;
    std::vector<wasm_type_t> local_types;

    void emit_binop(const char *mode, const SValue &a, const SValue &b);
};
