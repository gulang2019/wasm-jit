#include "SinglePassCompiler.h"
#include "utils.h"

#include "CodeReader.h"

#include <vector>
#include <unordered_map>
#include <limits.h>
#include <memory>
#include <set>
#include <fstream>



#define PER_VALUE_STACK_SIZE sizeof(value_t)

CodeGenerator::CodeGenerator(): 
Xbyak::CodeGenerator(4096), vfp(r9), tmp_i32(r8d), tmp_i16(r8w), tmp_i8(r8b), tmp_f64(xmm1), mem_start(r12), mem_end(r13), global_start(r14) {
    for (const Xbyak::Reg32* reg : { &edi, &esi, &r10d, &r11d, &r12d }) {
        i32_registers.push_back(Register(reg));    // Store by value, not reference
    }

    for (const Xbyak::Xmm* reg: {
        &xmm2, &xmm3, &xmm4, &xmm5, &xmm6, &xmm7
    }) {
        f64_registers.push_back(Register(reg));
    }
    TRACE("i32 registers:");
    for (Register& reg: i32_registers) {
        TRACE("%s ", reg.as_i32().toString());
    }
    TRACE("\n");
    
    TRACE("f64 registers:");
    for (Register& reg: f64_registers) {
        TRACE("%s ", reg.as_f64().toString());
    }
    TRACE("\n");
}

Register* CodeGenerator::alloc(wasm_type_t type) {
    auto& registers = (type == WASM_TYPE_I32) ? i32_registers : f64_registers;

    for (auto& reg: registers){
        if (reg.is_free()) {
            reg.last_used = timestamp++;
            TRACE("allocate %s %d\n", type == WASM_TYPE_I32? reg.as_i32().toString() : reg.as_f64().toString(), reg.is_i32);
            return &reg;
        }
    }

    // spill one register that is not used for a long time
    int min_last_used = INT_MAX;
    Register* spill_reg = nullptr;
    for (auto& reg: registers) {
        if (reg.last_used < min_last_used) {
            min_last_used = reg.last_used;
            spill_reg = &reg;
        }
    }
    spill_reg->last_used = timestamp++;
    if (!spill_reg->value->is_imm) { 
        if (type == WASM_TYPE_I32) {
            mov(ptr[vfp + spill_reg->value->offset], spill_reg->as_i32());
        }
        else {
            movsd(ptr[vfp + spill_reg->value->offset], spill_reg->as_f64());
        }
    }
    spill_reg->value->reg = nullptr;
    spill_reg->value = nullptr;
    return spill_reg;
}

Register* CodeGenerator::to_reg(AbstractValue* value) {
    if (value->reg) {
        assert(value->reg->value == value);
        return value->reg;
    }
    TRACE("Alloc  I32 %d\n", value->type == WASM_TYPE_I32);
    auto reg = alloc(value->type);
    reg->value = value;
    value->reg = reg;
    if (value->type == WASM_TYPE_I32) {
        if (value->is_imm) 
            mov(reg->as_i32(), value->imm.i32);
        else mov(reg->as_i32(), ptr[vfp + value->offset]);
    }
    else {
        assert(!value->is_imm);
        movsd(reg->as_f64(), ptr[vfp + value->offset]);
    }
    return reg;
}

AbstractValue* AbstractStack::push(wasm_type_t type, void* imm) {
    AbstractValue* value = new AbstractValue();
    values.push_back(value);
    value->type = type;
    if (imm) {
        value->is_imm = true;
        if (type == WASM_TYPE_I32) {
            value->imm.i32 = *(int32_t*)imm;
        } else if (type == WASM_TYPE_F64) {
            value->imm.f64 = *(double*)imm;
        }
    }
    else {
        value->is_imm = false;
    }
    value->offset = _offset;

    value->reg = nullptr; // we initialize the value to be in memory
    
    AbstractValueRef ref = {
        .value = value,
        .offset = _offset,
    };

    _offset += PER_VALUE_STACK_SIZE;
    _max_stack_offset = std::max(_max_stack_offset, _offset);
    stack.push_back(ref);
    value->refs.insert(stack.size() - 1);
    return value;
}

AbstractStack::~AbstractStack() {
    for (auto v: values) delete v;
}

void AbstractValue::print(std::ostream& o) const {
    if (type == WASM_TYPE_I32) {
        o << "i32 ";
        if (is_imm) {
            o << "imm " << imm.i32;
        }
        o << "reg: " << reg << " ";
        if (reg) {
            o << reg->as_i32().toString();
        }
        if (!is_imm and !reg) {
            o << "inMem " << offset;
        }
        o << std::endl;
    }
    if (type == WASM_TYPE_F64) {
        o << "f64 ";
        if (is_imm) {
            o << "imm " << imm.f64;
        }
        if (reg) {
            o << "reg " << reg->as_f64().toString();
        }
        if (!is_imm and !reg) {
            o << "inMem " << offset;
        }
        o << std::endl;
    }
}

void AbstractStack::print(std::ostream& o) const {
    o << "Stack" << std::endl;
    for (auto& ref: stack) {
        auto v = ref.value;
        v->print(o);
    }
    o << "Stack END" << std::endl;
}

size_t AbstractStack::size() {
    return stack.size();
}

void AbstractStack::push(AbstractValue* value) {
    AbstractValueRef ref = {
        .value = value,
        .offset = _offset,
    };

    _offset += PER_VALUE_STACK_SIZE;
    _max_stack_offset = std::max(_max_stack_offset, _offset);
    TRACE("pushing value %p\n", value);
    stack.push_back(ref);
    value->refs.insert(stack.size() - 1);
}

void trap() {
    printf("!trap\n");
    exit(0);
}

void AbstractStack::pop() {
    stack.back().value->refs.erase(stack.size() - 1);
    _offset -= PER_VALUE_STACK_SIZE;
    // if (!stack.back().value->refs.size()) {
    //     printf("delete value ");
    //     stack.back().value->print(std::cout);
    //     delete stack.back().value;
    // }
    stack.pop_back();
}

AbstractValue* AbstractStack::at(int idx) {
    if (idx < 0) {
        idx = stack.size() + idx;
    }
    return stack[idx].value;
}

void AbstractStack::write(size_t idx, AbstractValue* value) { 
    /**
        * write value to the stack at idx
        */
    auto& ref = stack[idx];
    ref.value->refs.erase(idx);
    if (!ref.value->refs.size()) {
        delete ref.value;
    }
    else if (ref.inMem() && ref.offset == ref.value->offset) {
        // we move the value to another reference
        auto reg = code_generator.to_reg(ref.value);
        if (ref.value->type == WASM_TYPE_I32) {
            code_generator.mov(reg->as_i32(), code_generator.ptr[code_generator.vfp + ref.offset]);
        }
        else if (ref.value->type == WASM_TYPE_F64) {
            code_generator.movsd(reg->as_f64(), code_generator.ptr[code_generator.vfp + ref.offset]);
        }
    }
    ref.value = value;
    value->refs.insert(idx);
}

void AbstractStack::flush_to_memory() {
    for (auto& ref: stack) {
        if (ref.inMem() && ref.offset == ref.value->offset)
            continue;
        if (ref.value->type == WASM_TYPE_I32) {
            if (ref.value->is_imm){
                code_generator.mov(code_generator.dword[code_generator.vfp + ref.offset], ref.value->imm.i32);
                ref.value->is_imm = false;
            }
            else if (ref.value->reg)
                code_generator.mov(code_generator.ptr[code_generator.vfp + ref.offset], ref.value->reg->as_i32());
            else {
                code_generator.mov(code_generator.tmp_i32, code_generator.ptr[code_generator.vfp + ref.value->offset]);
                code_generator.mov(code_generator.dword[code_generator.vfp + ref.offset], code_generator.tmp_i32);
            }
        }
        else if (ref.value->type == WASM_TYPE_F64) {
            assert(!ref.value->is_imm);
            if (ref.value->reg) // if the value is in register, we need to move it to memory
                code_generator.movsd(code_generator.ptr[code_generator.vfp + ref.offset], ref.value->reg->as_f64());
            else {
                code_generator.movsd(code_generator.tmp_f64, code_generator.ptr[code_generator.vfp + ref.value->offset]);
                code_generator.movsd(code_generator.ptr[code_generator.vfp + ref.offset], code_generator.tmp_f64);
            }
        }
    }
}

size_t AbstractStack::max_stack_offset() {
    return _max_stack_offset;
}

#define BINARY_CMP_I32(OP, TYPE, FUNC) {       \
    auto b = stack.at(-1); \
    stack.pop();            \
    auto a = stack.at(-1);  \
    stack.pop();            \
    if (a->is_imm && b->is_imm) {        \
        int value = int((*(TYPE*)(&a)) OP (*(TYPE*)(&b))); \
        stack.push(WASM_TYPE_I32, &value);\
    }\
    else {\
        auto a_reg = code_generator.to_reg(a);\
        auto b_reg = code_generator.to_reg(b);\
        auto d_reg = code_generator.to_reg(stack.push(WASM_TYPE_I32));\
        code_generator.cmp(a_reg->as_i32(), b_reg->as_i32());\
        code_generator.FUNC(code_generator.al);\
        code_generator.movzx(d_reg->as_i32(), code_generator.al);\
    }\
}

#define BINARY_OP_I32(OP, TYPE, FUNC) {       \
    auto b = stack.at(-1);\ 
    stack.pop();\            
    auto a = stack.at(-1);\
    stack.pop();\
    if (a->is_imm && b->is_imm) {\        
        int value = int((*(TYPE*)(&a)) OP (*(TYPE*)(&b)));\ 
        stack.push(WASM_TYPE_I32, &value);\
    }\
    else {\
        auto a_reg = code_generator.to_reg(a);\
        auto b_reg = code_generator.to_reg(b);\
        auto d_reg = code_generator.to_reg(stack.push(WASM_TYPE_I32));\
        code_generator.mov(d_reg->as_i32(), a_reg->as_i32());\
        code_generator.FUNC(d_reg->as_i32(), b_reg->as_i32());\
    }\
}

#define BINARY_OP_I32_SHIFT(FUNC) {       \
    auto b = stack.at(-1);\
    stack.pop();\  
    auto a = stack.at(-1);\
    stack.pop();\
    auto a_reg = code_generator.to_reg(a);\
    auto b_reg = code_generator.to_reg(b);\
    auto d_reg = code_generator.to_reg(stack.push(WASM_TYPE_I32));\
    code_generator.mov(d_reg->as_i32(), a_reg->as_i32());\
    code_generator.mov(code_generator.ecx, b_reg->as_i32());\
    code_generator.FUNC(d_reg->as_i32(), code_generator.cl);\
}

#define BINARY_OP_I32_AX(OP, TYPE, FUNC, IS_AX, XOR_EDX, EXT_EDX) {       \
    auto b = stack.at(-1); \
    stack.pop();            \
    auto a = stack.at(-1);  \
    stack.pop();            \
    if (a->is_imm && b->is_imm) {        \
        int value = int((*(TYPE*)(&a)) OP (*(TYPE*)(&b))); \
        stack.push(WASM_TYPE_I32, &value);\
    }\
    else {\
        auto a_reg = code_generator.to_reg(a);\
        auto b_reg = code_generator.to_reg(b);\
        auto d_reg = code_generator.to_reg(stack.push(WASM_TYPE_I32));\
        code_generator.mov(code_generator.eax, a_reg->as_i32());\
        if (XOR_EDX) code_generator.xor_(code_generator.edx, code_generator.edx);\
        if (EXT_EDX) code_generator.cdq();\
        code_generator.FUNC(b_reg->as_i32());\
        code_generator.mov(d_reg->as_i32(), IS_AX? code_generator.eax: code_generator.edx);\
    }\
}

#define F64_CMP(FUNC) {\
    auto b = stack.at(-1); \
    stack.pop();            \
    auto a = stack.at(-1);\
    stack.pop();\
    auto a_reg = code_generator.to_reg(a);\
    auto b_reg = code_generator.to_reg(b);\
    auto d_reg = code_generator.to_reg(stack.push(WASM_TYPE_I32));\
    code_generator.xor_(code_generator.eax, code_generator.eax);\
    code_generator.ucomisd(a_reg->as_f64(), b_reg->as_f64());\
    code_generator.FUNC(code_generator.al);\
    code_generator.movzx(d_reg->as_i32(), code_generator.al);\
}

#define BINARY_F64(FUNC) {\
    auto b = stack.at(-1);\ 
    stack.pop();\            
    auto a = stack.at(-1);\
    stack.pop();\
    auto a_reg = code_generator.to_reg(a);\
    auto b_reg = code_generator.to_reg(b);\
    auto d_reg = code_generator.to_reg(stack.push(WASM_TYPE_F64));\
    code_generator.movsd(d_reg->as_f64(), a_reg->as_f64());\
    code_generator.FUNC(d_reg->as_f64(), b_reg->as_f64());\
}

#define LOAD(TYPE, TYPETAG, MOV, WORD) {\
    auto offset = stack.at(-1);\
    stack.pop();\
    auto reg = code_generator.to_reg(offset);\
    auto mem_arg = codeptr.rd_mem_arg();\
    code_generator.xor_(code_generator.rax, code_generator.rax);\
    code_generator.mov(code_generator.eax, reg->as_i32());\
    code_generator.add(code_generator.rax, code_generator.mem_start);\
    code_generator.add(code_generator.rax, mem_arg.offset);\
    code_generator.cmp(code_generator.rax, code_generator.mem_end);\
    code_generator.jge("TRAP");\
    code_generator.cmp(code_generator.rax, code_generator.mem_start);\
    code_generator.jl("TRAP");\
    auto d_reg = code_generator.to_reg(stack.push(TYPE));\
    code_generator.MOV(d_reg->TYPETAG(), code_generator.WORD[code_generator.rax]);\
}

#define STORE \
    auto v = stack.at(-1);\
    auto v_reg = code_generator.to_reg(v);\
    stack.pop();\
    auto offset = stack.at(-1);\
    auto reg = code_generator.to_reg(offset);\
    stack.pop();\
    auto mem_arg = codeptr.rd_mem_arg();\
    code_generator.xor_(code_generator.rax, code_generator.rax);\
    code_generator.mov(code_generator.eax, reg->as_i32());\
    code_generator.add(code_generator.rax, code_generator.mem_start);\
    code_generator.add(code_generator.rax, mem_arg.offset);\
    code_generator.cmp(code_generator.rax, code_generator.mem_end);\
    code_generator.jge("TRAP");\
    code_generator.cmp(code_generator.rax, code_generator.mem_start);\
    code_generator.jl("TRAP");


#define STORE_I32(WORD, TMPREG){\
    STORE\
    code_generator.mov(code_generator.tmp_i32, v_reg->as_i32());\
    code_generator.mov(code_generator.WORD[code_generator.rax], code_generator.TMPREG);\
}

#define STORE_F64() {\
    STORE\
    code_generator.movsd(code_generator.qword[code_generator.rax], v_reg->as_f64());\
}

SinglePassCompiler::SinglePassCompiler(
    Function* func
): func(func), code_generator_ptr(std::make_unique<CodeGenerator>()), 
    stack(*code_generator_ptr) {}

struct SPCBlock{
    static size_t cnt;
    enum Type{
        BLOCK,
        LOOP,
    } type;
    std::string label;
    SPCBlock(Type _type): type(type) {
        label = "L" + std::to_string(SPCBlock::cnt++);
    }
};

size_t SPCBlock::cnt = 0;

void SinglePassCompiler::compile() {
    auto codeptr = CodePtr();
    codeptr.reset(&func->_decl.code_bytes, 0, func->_decl.code_bytes.size());
    
    auto& code_generator = *code_generator_ptr;

    // prologue, move the pointer to the virtual frame pointer
    code_generator.mov(code_generator.vfp, code_generator.rdi);
    code_generator.mov(code_generator.mem_start, code_generator.rsi);
    code_generator.mov(code_generator.mem_end, code_generator.rdx);
    code_generator.mov(code_generator.global_start, code_generator.rcx);

    for (auto t: func->_decl.sig->params) {
        stack.push(t);
    }

    TRACE("# inputs: %d\n", func->_decl.sig->params.size());
    TRACE("# outputs: %d\n", func->_decl.sig->results.size());
    std::vector<SPCBlock> blocks;

    int mem_acc_id = 0;

    blocks.emplace_back(SPCBlock::Type::BLOCK);
    while (!codeptr.is_end()) {
        auto opcode = codeptr.rd_opcode();
        switch (opcode) {
            case WASM_OP_UNREACHABLE:{
                code_generator.ud2();
                break;
            }
            case WASM_OP_NOP:{
                break;
            }
            case WASM_OP_BLOCK: {
                blocks.emplace_back(SPCBlock::Type::BLOCK);
                codeptr.skip_block_type();
                break;
            }
            case WASM_OP_LOOP:{
                blocks.emplace_back(SPCBlock::Type::LOOP);
                stack.flush_to_memory();
                code_generator.L(blocks.back().label.c_str());
                codeptr.skip_block_type();
                break;
            }
            case WASM_OP_IF:{
                ERR("WASM_OP_IF not implemented for single pass compiler");
                break;
            }
            case WASM_OP_ELSE:{
                ERR("WASM_OP_ELSE not implemented for single pass compiler");
                break;
            }
            case WASM_OP_END:{
                assert(blocks.size());
                if (blocks.size()) {
                    if (blocks.back().type == SPCBlock::Type::BLOCK){
                        stack.flush_to_memory();
                        code_generator.L(blocks.back().label.c_str());
                    }
                    blocks.pop_back();
                }

                if (codeptr.is_end()) {
                    _do_return();
                }
                break;
            }
            case WASM_OP_BR:{
                unsigned index = codeptr.rd_u32leb();
                stack.flush_to_memory();
                code_generator.jmp(blocks[blocks.size() - 1 - index].label);
                break;
            }
            case WASM_OP_BR_IF:{
                unsigned index = codeptr.rd_u32leb();
                auto v = stack.at(-1);
                stack.pop();
                auto reg = code_generator.to_reg(v);
                code_generator.mov(code_generator.eax, reg->as_i32());
                stack.flush_to_memory();
                code_generator.cmp(code_generator.eax, 0);
                code_generator.jne(blocks[blocks.size() - 1 - index].label);
                break;
            }
            case WASM_OP_BR_TABLE:{
                ERR("WASM_OP_BR_TABLE not implemented for single pass compiler");
                break;
            }
            case WASM_OP_RETURN:{
                _do_return();
                break;
            }
            case WASM_OP_CALL:{
                ERR("WASM_OP_CALL not implemented for single pass compiler");
                break;
            }
            case WASM_OP_CALL_INDIRECT:{
                ERR("WASM_OP_CALL_INDIRECT not implemented for single pass compiler");
                break;
            }
            case WASM_OP_DROP:{
                stack.pop();
                break;
            }
            case WASM_OP_SELECT:{
                ERR("WASM_OP_SELECT not implemented for single pass compiler");
                break;
            }
            case WASM_OP_LOCAL_GET:{
                stack.push(stack.at(codeptr.rd_i32leb()));
                break;
            }
            case WASM_OP_LOCAL_SET:{
                stack.write(codeptr.rd_i32leb(), stack.at(-1));
                stack.pop();
                break;
            }
            case WASM_OP_LOCAL_TEE:{
                stack.write(codeptr.rd_i32leb(), stack.at(-1));
                break;
            }
            case WASM_OP_GLOBAL_GET:{
                ERR("WASM_OP_GLOBAL_GET not implemented for single pass compiler");
                break;
            }
            case WASM_OP_GLOBAL_SET:{
                ERR("WASM_OP_GLOBAL_SET not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_LOAD:{
                LOAD(WASM_TYPE_I32, as_i32, mov, dword);
                break;
            }
            case WASM_OP_F64_LOAD:{
                LOAD(WASM_TYPE_F64, as_f64, movsd, qword);
                break;
            }
            case WASM_OP_I32_LOAD8_S:{
                LOAD(WASM_TYPE_I32, as_i32, movsx, byte);
                break;
            }
            case WASM_OP_I32_LOAD8_U:{
                LOAD(WASM_TYPE_I32, as_i32, movzx, byte);
                break;
            }
            case WASM_OP_I32_LOAD16_S:{
                LOAD(WASM_TYPE_I32, as_i32, movsx, word);
                break;
            }
            case WASM_OP_I32_LOAD16_U:{
                LOAD(WASM_TYPE_I32, as_i32, movzx, word);
                break;
            }
            case WASM_OP_I32_STORE:{
                STORE_I32(dword, tmp_i32);
                break;
            }
            case WASM_OP_F64_STORE:{
                STORE_F64();
                break;
            }
            case WASM_OP_I32_STORE8:{
                STORE_I32(byte, tmp_i8);
                break;
            }
            case WASM_OP_I32_STORE16:{
                STORE_I32(word, tmp_i16);
                break;
            }
            case WASM_OP_MEMORY_SIZE:{
                ERR("WASM_OP_MEMORY_SIZE not implemented for single pass compiler");
                break;
            }
            case WASM_OP_MEMORY_GROW:{
                ERR("WASM_OP_MEMORY_GROW not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_CONST:{
                auto value = codeptr.rd_i32leb();
                stack.push(WASM_TYPE_I32, &value);
                break;
            }
            case WASM_OP_F64_CONST:{
                ERR("WASM_OP_F64_CONST not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_EQZ:{
                int zero = 0;
                stack.push(WASM_TYPE_I32, &zero);
                BINARY_CMP_I32(==, int, sete);
                break;
            }
            case WASM_OP_I32_EQ:{
                /*
                var b = pop(), a = popReg();
                if (b.isConst()) asm.cmp_r_i(G(a.reg), b.const);
                else if (b.inReg()) asm.cmp_r_r(G(a.reg), G(b.reg));
                else asm.cmp_r_m(G(a.reg), A(masm.slotAddr(state.sp + 1)));
                var d = allocRegTos(ValueKind.I32), r1 = G(d);
                asm.set_r(cond, r1);
                asm.movbzx_r_r(r1, r1);
                state.push(KIND_I32 | IN_REG, d, 0);
                */
                BINARY_CMP_I32(==, int, sete);
                break;
            }
            case WASM_OP_I32_NE:{
                BINARY_CMP_I32(!=, int, setne);
                break;
            }
            case WASM_OP_I32_LT_S:{
                BINARY_CMP_I32(<, int, setl);
                break;
            }
            case WASM_OP_I32_LT_U:{
                BINARY_CMP_I32(<, unsigned, setb);
                break;
            }
            case WASM_OP_I32_GT_S:{
                BINARY_CMP_I32(>, int, setg)
                break;
            }
            case WASM_OP_I32_GT_U:{
                BINARY_CMP_I32(>, unsigned, seta);
                break;
            }
            case WASM_OP_I32_LE_S:{
                BINARY_CMP_I32(<=, int, setle);
                break;
            }
            case WASM_OP_I32_LE_U:{
                BINARY_CMP_I32(<=, unsigned, setbe);
                break;
            }
            case WASM_OP_I32_GE_S:{
                BINARY_CMP_I32(>=, int, setge);
                break;
            }
            case WASM_OP_I32_GE_U:{
                BINARY_CMP_I32(>=, unsigned, setae);
                break;
            }
            case WASM_OP_F64_EQ:{
                F64_CMP(sete);
                break;
            }
            case WASM_OP_F64_NE:{
                F64_CMP(setne);
                break;
            }
            case WASM_OP_F64_LT:{
                F64_CMP(setb);
                break;
            }
            case WASM_OP_F64_GT:{
                F64_CMP(seta);
                break;
            }
            case WASM_OP_F64_LE:{
                F64_CMP(setbe);
                break;
            }
            case WASM_OP_F64_GE:{
                F64_CMP(setae);
                break;
            }
            case WASM_OP_I32_CLZ:{
                ERR("WASM_OP_I32_CLZ not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_CTZ:{
                ERR("WASM_OP_I32_CTZ not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_POPCNT:{
                ERR("WASM_OP_I32_POPCNT not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_ADD:{
                BINARY_OP_I32(+, int, add);
                break;
            }
            case WASM_OP_I32_SUB:{
                BINARY_OP_I32(-, int, sub);
                break;
            }
            case WASM_OP_I32_MUL:{
                BINARY_OP_I32_AX(*, int, imul, true, false, false);
                break;
            }
            case WASM_OP_I32_DIV_S:{
                BINARY_OP_I32_AX(/, int, idiv, true, false, true);
                break;
            }
            case WASM_OP_I32_DIV_U:{
                BINARY_OP_I32_AX(/, unsigned, div, true, true, false);
                break;
            }
            case WASM_OP_I32_REM_S:{
                BINARY_OP_I32_AX(/, int, idiv, false, false, true);
                break;
            }
            case WASM_OP_I32_REM_U:{
                BINARY_OP_I32_AX(/, unsigned, div, false, true, false);
                break;
            }
            case WASM_OP_I32_AND:{
                BINARY_OP_I32(&, int, and_);
                break;
            }
            case WASM_OP_I32_OR:{
                BINARY_OP_I32(|, int, or_);
                break;
            }
            case WASM_OP_I32_XOR:{
                BINARY_OP_I32(^, int, xor_);
                break;
            }
            case WASM_OP_I32_SHL:{
                BINARY_OP_I32_SHIFT(shl);
                break;
            }
            case WASM_OP_I32_SHR_S:{
                BINARY_OP_I32_SHIFT(sar);
                break;
            }
            case WASM_OP_I32_SHR_U:{
                BINARY_OP_I32_SHIFT(shr);
                break;
            }
            case WASM_OP_I32_ROTL:{
                BINARY_OP_I32_SHIFT(rol);
                break;
            }
            case WASM_OP_I32_ROTR:{
                BINARY_OP_I32_SHIFT(ror);
                break;
            }
            case WASM_OP_F64_ABS:{
                ERR("WASM_OP_F64_ABS not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_NEG:{
                ERR("WASM_OP_F64_NEG not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_CEIL:{
                ERR("WASM_OP_F64_CEIL not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_FLOOR:{
                ERR("WASM_OP_F64_FLOOR not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_TRUNC:{
                ERR("WASM_OP_F64_TRUNC not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_NEAREST:{
                ERR("WASM_OP_F64_NEAREST not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_SQRT:{
                ERR("WASM_OP_F64_SQRT not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_ADD:{
                BINARY_F64(addsd);
                break;
            }
            case WASM_OP_F64_SUB:{
                BINARY_F64(subsd);
                break;
            }
            case WASM_OP_F64_MUL:{
                BINARY_F64(mulsd);
                break;
            }
            case WASM_OP_F64_DIV:{
                BINARY_F64(divsd);
                break;
            }
            case WASM_OP_F64_MIN:{
                BINARY_F64(minsd);
                break;
            }
            case WASM_OP_F64_MAX:{
                BINARY_F64(maxsd);
                break;
            }
            case WASM_OP_I32_TRUNC_F64_S:{
                ERR("WASM_OP_I32_TRUNC_F64_S not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_TRUNC_F64_U:{
                ERR("WASM_OP_I32_TRUNC_F64_U not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_CONVERT_I32_S:{
                ERR("WASM_OP_F64_CONVERT_I32_S not implemented for single pass compiler");
                break;
            }
            case WASM_OP_F64_CONVERT_I32_U:{
                ERR("WASM_OP_F64_CONVERT_I32_U not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_EXTEND8_S:{
                ERR("WASM_OP_I32_EXTEND8_S not implemented for single pass compiler");
                break;
            }
            case WASM_OP_I32_EXTEND16_S:{
                ERR("WASM_OP_I32_EXTEND16_S not implemented for single pass compiler");
                break;
            }
        };
    }

    code_generator.L("TRAP");
    code_generator.mov(code_generator.rax, reinterpret_cast<uintptr_t>(&trap));
    code_generator.call(code_generator.rax);
    
    std::ofstream outFile("generated_code.bin", std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(code_generator.getCode()), code_generator.getSize());
    TRACE("generated code %p Executable %d\n", code_generator.getCode(), check_executable(code_generator.getCode()));
    func->code_generator = std::move(code_generator_ptr);
    func->max_stack_offset = stack.max_stack_offset();
}

void SinglePassCompiler::_do_return() {
    assert(stack.size() == func->_decl.sig->results.size() + func->_decl.sig->params.size());
    assert(func->_decl.sig->results.size() == 1);

    auto v = stack.at(-1);
    if (func->_decl.sig->results.front() == WASM_TYPE_I32) {
        if (v->is_imm) {
            code_generator_ptr->mov(code_generator_ptr->eax, v->imm.i32);
        }
        else if (v->reg) {
            code_generator_ptr->mov(code_generator_ptr->eax, v->reg->as_i32());
        }
        else {
            code_generator_ptr->mov(code_generator_ptr->eax, code_generator_ptr->ptr[code_generator_ptr->vfp + v->offset]);
        }
    }
    else {
        assert(func->_decl.sig->results.front() == WASM_TYPE_F64);
        if (v->reg) {
            code_generator_ptr->movsd(code_generator_ptr->xmm0, v->reg->as_f64());
        }
        else {
            code_generator_ptr->movsd(code_generator_ptr->xmm0, code_generator_ptr->ptr[code_generator_ptr->vfp + v->offset]);
        }
    }
    code_generator_ptr->ret();
}