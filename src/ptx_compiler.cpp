#include "ptx_compiler.h"

#include "CodeReader.h"

void PTXCompiler::compile(const FuncDecl &func) {
    TRACE("Start of PTX compile\n");
    auto codeptr = CodePtr();
    codeptr.reset(&func.code_bytes, 0, func.code_bytes.size());
    n_locals = 0;
    local_types.clear();

    for (auto p: func.sig->params) {
        stack.push(SValue(n_locals, p));
        local_types.push_back(p);
        n_locals++;
    }
    for (auto i = 0; i < func.num_pure_locals; i++) {
        stack.push(SValue(n_locals, WASM_TYPE_I32));
        local_types.push_back(WASM_TYPE_I32);
        n_locals++;
    }

    TRACE("# inputs: %lu\n", func.sig->params.size());
    TRACE("# outputs: %lu\n", func.sig->results.size());

    // header
    masm.gen_headers();
    masm.gen_func_start("test_func", func);

    for (auto i = func.sig->params.size(); i < n_locals; i++) {
        masm.gen_local(i);
    }

    auto scratch_a = masm.gen_scratch_reg(n_locals);
    auto scratch_b = masm.gen_scratch_reg(n_locals);

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
                auto from_reg = masm.reg(stack.pop(), scratch_a);

                // assert same type

            }

            default: {
                ERR("Unimplemented opcode [0x%x]\n", code);
                break;
            }
        }
    }

    masm.gen_func_finish();

    TRACE("\n\n\n=== OUTPUT ===\n\n");
    std::cout << masm.build() << std::endl;
}
