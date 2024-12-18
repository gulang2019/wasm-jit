#include "ptx_compiler.h"
#include "CodeReader.h"

struct AbstractBlock {
    std::string label;
    std::vector<SValue>* snapshot = nullptr;
    ~AbstractBlock() {
        if (snapshot) {
            free(snapshot);
        }
    }
};

void PTXCompiler::compile(
    const char* fn_name,
    const FuncDecl &func) {
    TRACE("Start of PTX compile\n");
    auto codeptr = CodePtr();
    codeptr.reset(&func.code_bytes, 0, func.code_bytes.size());

    // header
    masm.gen_headers();
    masm.gen_func_start(fn_name, func, stack);
    stack.end_of_params();

    for (auto k = 0; k < func.num_pure_locals; k++) {
        stack.push(WASM_TYPE_I32);
    }

    TRACE("# inputs: %lu\n", func.sig->params.size());
    TRACE("# outputs: %lu\n", func.sig->results.size());

    std::vector<AbstractBlock> blocks;
    int n_blocks = 0;

    while (!codeptr.is_end()) {
        auto code = codeptr.rd_opcode();

        // masm.emit_comment(opcode_table[code].mnemonic); 

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

            case WASM_OP_LOOP: {
                codeptr.skip_block_type();
                blocks.push_back({});
                blocks.back().label = "$L" + std::to_string(n_blocks++);
                blocks.back().snapshot = nullptr;
                masm.emit_label(blocks.back().label);
                stack.canonicalize(masm, blocks.back().snapshot);
                break;
            }

            case WASM_OP_BR_IF: {
                unsigned index = codeptr.rd_u32leb();
                auto& block = blocks[blocks.size() - 1 - index];
                auto v = stack.pop();
                stack.canonicalize(masm, block.snapshot);
                masm.emit_branch(v, block.label);
                break;
            }

            case WASM_OP_I32_LT_S: {
                emit_binop("setp.lt", WASM_TYPE_PREDICATE);
                break;
            }

            case WASM_OP_I32_ADD: {
                emit_binop("add", WASM_TYPE_I32);
                break;
            }

            case WASM_OP_I32_DIV_S: {
                emit_binop("div", WASM_TYPE_I32);
                break;
            }

            case WASM_OP_I32_SUB: {
                emit_binop("sub", WASM_TYPE_I32);
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

            case WASM_OP_END: { 
                if (blocks.size()) {
                    blocks.pop_back();
                }
                 break; 
            }

            case WASM_OP_F64_ADD: {
                emit_binop("add", WASM_TYPE_F64);
                break;
            }
            
            case WASM_OP_F64_MUL: {
                emit_binop("mul", WASM_TYPE_F64);
                break;
            }

            case WASM_OP_F64_CONST: {
                auto& r = stack.push(WASM_TYPE_F64);
                auto imm = codeptr.rd_u64();
                masm.emit_mov_f64(r, *reinterpret_cast<double*>(&imm));
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
                ERR("Unimplemented opcode [0x%x] %s\n", code, opcode_table[code].mnemonic);
                masm.emit_unknown(code);
                break;
            }
        }
    }

    masm.gen_finish(stack);
   
}

void PTXCompiler::emit_binop(const char *mode, 
                            wasm_type_t res_type) {
    auto src_b = stack.pop();
    auto src_a = stack.pop();
    auto& v = stack.push(res_type);
    masm.emit_binop(mode, v, src_a, src_b);
}

void PTXCompiler::emit(std::ostream& out) {
    TRACE("\n\n\n=== OUTPUT ===\n\n");
    out << masm.build(stack);
}