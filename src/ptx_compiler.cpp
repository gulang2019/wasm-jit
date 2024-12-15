#include "ptx_compiler.h"

#include "CodeReader.h"

void PTXCompiler::compile(const FuncDecl &func) {
    TRACE("Start of PTX compile\n");
    auto codeptr = CodePtr();
    codeptr.reset(&func.code_bytes, 0, func.code_bytes.size());

    // header
    masm.gen_headers();
    masm.gen_func_start("test_func", func);

    reg_t i = 0;
    for (auto p: func.sig->params) {
        stack.push(SValue(i++, p));
    }
    for (auto k = 0; k < func.num_pure_locals; k++) {
        masm.gen_local();
        stack.push(SValue(i++, WASM_TYPE_I32));
    }

    TRACE("# inputs: %lu\n", func.sig->params.size());
    TRACE("# outputs: %lu\n", func.sig->results.size());

    while (!codeptr.is_end()) {
        auto code = codeptr.rd_opcode();

        switch (code) {
            case WASM_OP_UNREACHABLE:
            case WASM_OP_NOP: break;

            case WASM_OP_LOCAL_GET: {
                stack.push(stack.at(codeptr.rd_i32leb()));
                break;
            }

            case WASM_OP_LOCAL_SET: {
                auto to = stack.at(codeptr.rd_i32leb());
                auto from = stack.pop();

                masm.emit_mov(to.local, from);
                break;
            }

            default: {
                ERR("Unimplemented opcode [0x%x]\n", code);
                masm.emit_unknown(code);
                break;
            }
        }
    }

    masm.gen_func_finish();

    TRACE("\n\n\n=== OUTPUT ===\n\n");
    std::cout << masm.build() << std::endl;
}
