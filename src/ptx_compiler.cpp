#include "ptx_compiler.h"

#include "CodeReader.h"

void PTXCompiler::compile(const FuncDecl *func) {
    TRACE("Start of PTX compile\n");
    auto codeptr = CodePtr();
    codeptr.reset(&func->code_bytes, 0, func->code_bytes.size());
    local_types.clear();

    int n_locals = 0;
    for (auto p: func->sig->params) {
        stack.push(SValue(n_locals, p));
        local_types.push_back(p);
        n_locals++;
    }
    for (auto i = 0; i < func->num_pure_locals; i++) {
        stack.push(SValue(n_locals, WASM_TYPE_I32));
        local_types.push_back(WASM_TYPE_I32);
        n_locals++;
    }

    TRACE("# inputs: %lu\n", func->sig->params.size());
    TRACE("# outputs: %lu\n", func->sig->results.size());

    // header
    masm.gen_headers();

    while (!codeptr.is_end()) {
        auto code = codeptr.rd_opcode();

        switch (code) {
            case WASM_OP_UNREACHABLE:
            case WASM_OP_NOP: break;

            default: {
                ERR("Unimplemented opcode [0x%x]\n", code);
                return;
            }
        }
    }
}
