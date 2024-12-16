#include "ptx_compiler.h"

#include "CodeReader.h"

void PTXCompiler::compile(
    const char* fn_name,
    const FuncDecl &func) {
    TRACE("Start of PTX compile\n");
    auto codeptr = CodePtr();
    codeptr.reset(&func.code_bytes, 0, func.code_bytes.size());

    // header
    masm.gen_headers();
    masm.gen_func_start(fn_name, func, stack);

    for (auto k = 0; k < func.num_pure_locals; k++) {
        stack.push(WASM_TYPE_I32);
    }

    TRACE("# inputs: %lu\n", func.sig->params.size());
    TRACE("# outputs: %lu\n", func.sig->results.size());

    while (!codeptr.is_end()) {
        auto code = codeptr.rd_opcode();

        masm.emit_comment(opcode_table[code].mnemonic); 

        switch (code) {
            case WASM_OP_UNREACHABLE:
            case WASM_OP_NOP: break;

            case WASM_OP_LOCAL_GET: {
                stack.push(stack.at(codeptr.rd_i32leb()));
                break;
            }

            case WASM_OP_LOCAL_SET: {
                auto v = stack.pop();
                stack.set(codeptr.rd_i32leb(), v);
                break;
            }

            case WASM_OP_I32_MUL: {
                emit_binop("mul.lo", WASM_TYPE_I32);
                break;
            }

            case WASM_OP_I32_ADD: {
                emit_binop("add", WASM_TYPE_I32);
                break;
            }

            case WASM_OP_I32_CONST: {
                auto& r = stack.push(WASM_TYPE_I32);
                auto v = codeptr.rd_i32leb();
                masm.emit_mov_i32(r, v);
                break;
            }

            case WASM_OP_I32_LOAD: {
                auto mem_arg = codeptr.rd_mem_arg();
                auto offset = stack.pop();
                auto& r = stack.push(WASM_TYPE_I32);
                masm.emit_load("global", r, offset);
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

            case WASM_OP_F64_ADD: {
                emit_binop("add", WASM_TYPE_F64);
                break;
            }

            case WASM_OP_F64_LOAD: {
                auto mem_arg = codeptr.rd_mem_arg();
                auto offset = stack.pop();
                auto& r = stack.push(WASM_TYPE_F64);
                masm.emit_load("global", r, offset);
                break;
            }

            case WASM_OP_F64_STORE: {
                auto mem_arg = codeptr.rd_mem_arg();
                auto v = stack.pop();
                auto addr = stack.pop();
                masm.emit_store("global", addr, v);
                break;
            }

            default: {
                ERR("Unimplemented opcode [0x%x]\n", code);
                masm.emit_unknown(code);
                break;
            }
        }
    }

    masm.gen_finish(stack);
   
}

void PTXCompiler::emit_binop(const char *mode, 
                            wasm_type_t res_type) {
    auto lval = stack.pop();
    auto rval = stack.pop();
    auto& v = stack.push(res_type);
    masm.emit_binop(mode, v, lval, rval);
}

void PTXCompiler::emit(std::ostream& out) {
    TRACE("\n\n\n=== OUTPUT ===\n\n");
    out << masm.build(stack);
}