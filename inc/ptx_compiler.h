#pragma once
#include <sstream>

#include "ptx_assmbler.h"
#include "SinglePassCompiler.h"
#include "ir.h"

class PTXCompiler {

public:
    void compile(const char* fn_name, const FuncDecl &f);
    void emit(std::ostream& out);

private:
    PtxAsm masm;
    PtxStack stack;
    std::vector<wasm_type_t> local_types;

    void emit_binop(const char *mode, wasm_type_t type);
};
