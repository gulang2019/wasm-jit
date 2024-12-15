#pragma once

#include "ptx_assmbler.h"
#include "SinglePassCompiler.h"
#include "ir.h"

constexpr size_t N_ENV = 0;
constexpr const char *ENV_PARAMS[] = {
    // TODO: this
};

class PTXCompiler {

public:
    void compile(const FuncDecl &f);

private:
    PtxAsm masm;
    PtxStack stack;
    std::vector<wasm_type_t> local_types;
};
