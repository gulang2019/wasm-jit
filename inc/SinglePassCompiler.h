#pragma once 
#include "instance.h"

#include <set>
#include <iostream>

struct AbstractValue;

struct Register {
    bool is_i32;
    const void* reg;
    AbstractValue* value;
    int last_used;

    // Constructor for Reg32 (integer register)
    Register(const Xbyak::Reg32* _reg) : is_i32(true), reg(_reg), value(nullptr), last_used(0) {
    }

    // Constructor for Xmm (floating-point register)
    Register(const Xbyak::Xmm* _reg) : is_i32(false), reg(_reg), value(nullptr), last_used(0) {
    }

    const Xbyak::Reg32& as_i32() {
        assert(is_i32);
        return *(Xbyak::Reg32*)reg;
    }

    const Xbyak::Xmm& as_f64() {
        assert(!is_i32);
        return *(Xbyak::Xmm*)reg;
    }

    bool is_free() {
        return value == nullptr;
    }
    
};

struct AbstractValue {
    /*
    An abstract value always backed by a memory position, 
    but it can be stored in a register or a pure constant
    */
    wasm_type_t type;
    
    /** fields when */
    Register* reg;
    size_t offset; // imm do not need this
    std::set<int> refs;

    bool is_imm;
    
    union IMM {
        int32_t i32;
        double f64;
    }imm;

    void print(std::ostream& o) const;

};

struct AbstractValueRef {
    AbstractValue* value;
    size_t offset;
    bool inMem() {
        return value->reg == nullptr and !value->is_imm;
    }
};

struct CodeGenerator: public Xbyak::CodeGenerator {
    const Xbyak::Reg64& vfp;
    const Xbyak::Reg32& tmp_i32;
    const Xbyak::Reg16& tmp_i16;
    const Xbyak::Reg8& tmp_i8;
    const Xbyak::Reg64& mem_start;
    const Xbyak::Reg64& mem_end;
    const Xbyak::Reg64& global_start;
    const Xbyak::Xmm& tmp_f64;
    std::vector<Register> i32_registers;
    std::vector<Register> f64_registers;
    size_t timestamp = 0; 
public:
    CodeGenerator();
    Register* alloc(wasm_type_t type);
    Register* to_reg(AbstractValue* value);
};

class AbstractStack {
    std::vector<AbstractValueRef> stack;
    std::vector<AbstractValue*> values;
    size_t _offset = 0;
    size_t _max_stack_offset = 0;
    CodeGenerator& code_generator;
public: 
    AbstractStack(CodeGenerator& code_generator): code_generator(code_generator) {}
    ~AbstractStack();
    AbstractValue* push(wasm_type_t type, void* imm = nullptr);
    void push(AbstractValue* value);
    void pop();
    AbstractValue* at(int idx);
    void write(size_t idx, AbstractValue* value);
    void flush_to_memory();
    size_t max_stack_offset();
    size_t size();
    void print(std::ostream& o) const; 
};

class SinglePassCompiler {
    Function* func; 
    std::unique_ptr<CodeGenerator> code_generator_ptr;
    AbstractStack stack;
public: 
    SinglePassCompiler(Function * func);
    void compile(); 
    void _do_return();
};

