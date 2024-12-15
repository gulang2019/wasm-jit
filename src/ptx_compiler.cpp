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
                auto v = stack.at(codeptr.rd_i32leb());
                auto r = masm.new_reg(v);
                stack.push(v.with_reg(r));
                break;
            }

            case WASM_OP_LOCAL_SET: {
                auto to = stack.at(codeptr.rd_i32leb());
                auto from = stack.pop();
                masm.emit_mov(to.local, from);
                break;
            }

            case WASM_OP_I32_MUL: {
                auto b = stack.pop(), a = stack.pop();
                emit_binop("mul.lo", a, b);
                break;
            }

            case WASM_OP_I32_ADD: {
                auto b = stack.pop(), a = stack.pop();
                emit_binop("add", a, b);
                break;
            }

            case WASM_OP_I32_CONST: {
                auto v = codeptr.rd_i32leb();
                auto r = masm.gen_local();
                masm.emit_mov_i32(r, v);
                stack.push(SValue(r, WASM_TYPE_I32));
                break;
            }

            case WASM_OP_I32_LOAD: {
                auto mem_arg = codeptr.rd_mem_arg();
                auto v = stack.pop();
                const char *mode;
                if (v.local < func.sig->params.size()) mode = "param";
                else mode = "global";
                auto r = masm.gen_local();
                masm.emit_load(mode, r, v);
                break;
            }

            case WASM_OP_I32_STORE: {
                auto mem_arg = codeptr.rd_mem_arg();
                auto v = stack.pop();
                auto addr = stack.pop();
                masm.emit_store("global", addr, v);
                break;
            }

            case WASM_OP_END: { break; }

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

void PTXCompiler::emit_binop(const char *mode, const SValue &a, const SValue &b) {
    reg_t dest = masm.gen_local();
    masm.emit_binop(mode, dest, a, b);
    stack.push(a.with_reg(dest));
}
