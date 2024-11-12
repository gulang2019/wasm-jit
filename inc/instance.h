#pragma once
#include <stdio.h>
#include <unordered_map>
#include <memory>

#include "ir.h"
#include "xbyak/xbyak.h"

class Instance; 

struct Throwable {
    const char* reason;
    const char* file;
    int lineno;
};


union value_t{
        int32_t i32;
        unsigned int u32;
        int64_t i64;
        unsigned long u64;
        float f32;
        double f64;
};

struct Value {
    Value(wasm_type_t type, void* data);
    Value(wasm_type_t type, const std::string& data);
    Value(unsigned int u32): _type(WASM_TYPE_I32) {_value.u32 = u32;}
    Value(int32_t i32): _type(WASM_TYPE_I32) {_value.i32 = i32;}
    Value(int64_t i64): _type(WASM_TYPE_I64) {_value.i64 = i64;}
    Value(unsigned long u64): _type(WASM_TYPE_I64) {_value.u64 = u64;}
    Value(float f32): _type(WASM_TYPE_F32) {_value.f32 = f32;}
    Value(double f64): _type(WASM_TYPE_F64) {_value.f64 = f64;}
    void print(FILE* file = stdout);
    wasm_type_t _type;
    value_t _value;
};

struct Memory {
    Memory(MemoryDecl& decl, std::list<DataDecl>& datas);
    void* at(unsigned offset, unsigned size, Throwable* throwable);
    unsigned size() {return _data.size();}
    int grow(int delta);
    MemoryDecl& _decl;
    bytearr _data;
    int _n_pages;
};
  
typedef std::unordered_map<int, int> j_table_t;

struct Function {
    Function(FuncDecl& decl, Instance& instance);
    FuncDecl& _decl; 
    Instance& _instance;
    j_table_t _j_table;
    std::unique_ptr<Xbyak::CodeGenerator> code_generator;
    size_t max_stack_offset;
    std::vector<double> f64_imms;
};

// struct Sig{
//     Sig(SigDecl& decl): _decl(decl) {}
//     SigDecl& _decl;
// };

struct Global {
    Global(GlobalDecl& decl);
    GlobalDecl& _decl;
    Value _value;
};

struct Table {
    Table(TableDecl& decl, std::list<ElemDecl>& elems);
    TableDecl& _decl;
    std::vector<FuncDecl*> _funcs;
    FuncDecl* get_func(int index) {
        return _funcs.at(index);
    }
};

struct Data {
    Data(DataDecl& decl): _decl(decl) {}
    DataDecl& _decl;
};

struct Elem {
    Elem(ElemDecl& decl): _decl(decl) {}
    ElemDecl& _decl;
};

struct MemArg {
    unsigned flags;
    unsigned memory_index;
    unsigned long offset;
};

/*
class Instance(module: Module, imports: Array<Exportable>) {
	def memories = Array<Memory>.new(module.memories.length);
	def functions = Array<Function>.new(module.functions.length);
	def sig_ids = Array<int>.new(module.heaptypes.length);
	def tags = Array<Tag>.new(module.tags.length);
	def globals = Array<Global>.new(module.globals.length);
	def tables = Array<Table>.new(module.tables.length);
	def exports = Array<Exportable>.new(module.exports.length);
}
*/
class Instance {
public:
    Instance(WasmModule& module);
    WasmModule& _module;
    std::vector<Memory> _memories;
    std::vector<Function> _functions;
    // std::vector<int> _sig_ids;
    // std::vector<Tag> _tags;
    std::vector<Global> _globals;
    std::vector<Table> _tables;
    // std::vector<Exportable> _exports;
};

// class Instantiator {
// public: 
//     Instantiator(
//         WasmModule& module
//     ): _module(module) {}

//     void run();
// private:
//     WasmModule& _module;
// };