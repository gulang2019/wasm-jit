#include <cassert>
#include <string>
#include <cstring>

#include "instance.h"
#include "CodeReader.h"
#include "SinglePassCompiler.h"

Value::Value(wasm_type_t type, void* data): _type(type) {
    if (data == nullptr){
        return;
    }
    switch (type){
        case WASM_TYPE_I32:
            _value.i32 = *(int32_t*)data;
            break;
        case WASM_TYPE_I64:
            _value.i64 = *(int64_t*)data;
            break;
        case WASM_TYPE_F32:
            _value.f32 = *(float*)data;
            break;
        case WASM_TYPE_F64:
            _value.f64 = *(double*)data;
            break;
        default:
            assert(!"Invalid type");
    }
}

Value::Value(wasm_type_t type, const std::string& data): _type(type) {
    switch (type){
        case WASM_TYPE_I32:
            _value.i32 = std::stoi(data);
            break;
        case WASM_TYPE_I64:
            _value.i64 = std::stoll(data);
            break;
        case WASM_TYPE_F32:
            _value.f32 = std::stof(data);
            break;
        case WASM_TYPE_F64:
            _value.f64 = std::stod(data);
            break;
        default:
            assert(!"Invalid type");
    }
}

void Value::print(FILE* file) {
    switch(_type){
      case WASM_TYPE_I32:
      fprintf(file, "%d\n", _value.i32);
      break;
      case WASM_TYPE_I64:
      fprintf(file, "%ld\n", _value.i64);
      break;
      case WASM_TYPE_F32:
      fprintf(file, "%.6f\n", _value.f32);
      break;
      case WASM_TYPE_F64:
      fprintf(file, "%.6f\n", _value.f64);
      break;
      default: 
      ERR("no return values");
    }
}

Memory::Memory(MemoryDecl& decl, std::list<DataDecl>& datas): _decl(decl) {
    _data.resize(_decl.limits.initial << 16);
    for (auto& data: datas) {
        if (&decl == data.mem){
            TRACE("Cpy data into memory: %d %d\n", data.mem_offset, data.bytes.size());
            assert(data.mem_offset + data.bytes.size() <= _data.size());
            memcpy(_data.data() + data.mem_offset, data.bytes.data(), data.bytes.size());
        }
    }
    TRACE("Memory size: %d\n", _data.size());
}

void* Memory::at(unsigned offset, unsigned size, Throwable* throwable) {
    TRACE("Memory::at %u %u %u\n", offset, size, offset + size);
    if ((offset + size) > _data.size() || offset >= _data.size()){
        TRACE("Memory out of bounds access %d > %d\n", offset + size, _data.size());
        if (throwable){
            throwable->reason = "out of bounds memory access";
            throwable->file = __FILE__;
            throwable->lineno = __LINE__;
        }
        return nullptr;
    }
    return _data.data() + offset;
}

int Memory::grow(int delta) {
    assert (delta >= 0);
    if (_n_pages + delta > _decl.limits.max){
        return -1;
    }
    _data.resize(_data.size() + delta << 16);
    _n_pages += delta;
    return _n_pages - delta;
}

Global::Global(GlobalDecl& decl) : 
    _decl(decl), 
    _value(decl.type, decl.init_expr_bytes.data()) {}

Instance::Instance(WasmModule& module): _module(module) {
    for (auto& mem: _module.Memories()){
        _memories.push_back(Memory(mem, _module.Datas()));
    }
    for (auto &global: _module.Globals()){
        _globals.push_back(Global(global));
    }
    for (auto &func: _module.Funcs()){
        _functions.push_back(Function(func, *this));
    }
    for (auto &table: _module.Tables()){
        _tables.push_back(Table(table, _module.Elems()));
    }

    assert(_module.get_imports().get_num_imports() == 0);
    
    TRACE("Instance created\n");
    TRACE("# of memories: %d\n", _memories.size());
    TRACE("# of functions: %d\n", _functions.size());
    TRACE("# of globals: %d\n", _globals.size());
    TRACE("# of tables: %d\n", _tables.size());
    TRACE("# of sigs: %d\n", _module.Sigs().size());
    TRACE("# of elements: %d\n", _module.Elems().size());
    TRACE("# of datas: %d\n", _module.Datas().size());
    /*
    
struct DataDecl {
  uint32_t flag;
  Opcode_t opcode_offset;
  uint32_t mem_offset;
  MemoryDecl *mem;
  bytearr bytes;
};
    */
    // for (auto& data: _module.Datas()){
    //     TRACE("Data: %d\n", data.flag);
    //     TRACE("Data: %s\n", opcode_table[data.opcode_offset].mnemonic);
    //     TRACE("Data: %d\n", data.mem_offset);
    // }

};

struct Block{
    int start;
    enum {LOOP, BLOCK, IF} type;
    std::vector<int> j_here; // positions that jump to either the start or End for this block
};

void parse_j_table(FuncDecl* func, j_table_t& j_table) {
    auto codeptr = CodePtr();
    codeptr.reset(&func->code_bytes, 0, func->code_bytes.size());
    std::vector<Block> blocks;
    blocks.push_back({0,Block::BLOCK}); // for entering the function
    while (!codeptr.is_end()) {
        auto pos = codeptr.offset();
        auto opcode = codeptr.rd_opcode();

        opcode_entry_t entry = opcode_table[opcode];

        unsigned index = -1;
        if (opcode == WASM_OP_BR || opcode == WASM_OP_BR_IF) {
            index = codeptr.rd_u32leb();
        }
        else {
            switch(entry.imm_type) {
                case IMM_NONE: {
                    break;
                }
                case IMM_BLOCKT: {
                    codeptr.skip_block_type();
                    break;
                }
                case IMM_LABELS: {
                    codeptr.rd_labels();
                    break;
                }
                case IMM_SIG_TABLE: {
                    codeptr.rd_u32leb();
                    codeptr.rd_u32leb();
                    break;
                }
                case IMM_FUNC:
                case IMM_LABEL:
                case IMM_LOCAL: 
                case IMM_GLOBAL:
                case IMM_I32:
                case IMM_MEMORY:
                case IMM_F32: {
                    codeptr.rd_u32leb();
                    break;
                }
                case IMM_TABLE: {
                    ERR("IMM_TABLE not implemented");
                    break;
                }
                case IMM_MEMARG: {
                    codeptr.rd_mem_arg();
                    break;
                }
                case IMM_F64:
                case IMM_I64: {
                    codeptr.rd_u64();
                    break;
                }
                default: 
                    ERR("opcode not implemented %s", entry.mnemonic);
            }; 
        }
        auto pos_after = codeptr.offset();

        switch (opcode) {
            case WASM_OP_IF: {
                blocks.push_back({pos, Block::IF});
                blocks.back().j_here.push_back(pos_after); // hack on the j_here to jump to the false branch
                break;
            }
            case WASM_OP_BLOCK:{
                blocks.push_back({pos, Block::BLOCK});
                break;
            }
            case WASM_OP_LOOP:{
                blocks.push_back({pos, Block::LOOP});
                break;
            }
            case WASM_OP_ELSE:{
                // the false branch of if should go here
                assert(blocks.back().type == Block::IF);
                assert(blocks.back().j_here.size() == 1);
                j_table[blocks.back().j_here.back()] = pos_after; // the false branch jumps here; 
                blocks.back().j_here.pop_back();
                blocks.back().j_here.push_back(pos_after); // the else branch should jump to the end;
                break;
            }
            case WASM_OP_BR: 
            case WASM_OP_BR_IF: {
                assert(index != -1);
                assert(blocks.size() > index);
                blocks[blocks.size()-1-index].j_here.push_back(pos_after);
                break;
            }
            case WASM_OP_END: {
                auto& block = blocks.back();
                for (auto j: block.j_here){
                    j_table[j] = block.type == Block::LOOP? block.start:pos;
                }
                blocks.pop_back();
                break;
            }
            default: 
                break;
        };
        
        // printf("opcode: %s %d %d\n", entry.mnemonic, pos, pos_after);
    }
    if (g_trace) {
        for (auto& j: j_table){
            TRACE("j_table[%d] = %d\n", j.first, j.second);
        }   
    }
    // exit(0);
}

Function::Function(FuncDecl& decl, Instance& instance): _decl(decl), _instance(instance) {
    parse_j_table(&_decl, _j_table);
    SinglePassCompiler(this).compile();
}

Table::Table(TableDecl& decl, std::list<ElemDecl>& elems):
_decl(decl){
    _funcs.resize(decl.limits.initial, nullptr);
    for (auto& elem: elems){
        int idx = 0;
        for (auto i: elem.func_indices){
            assert(elem.table_offset + idx < _funcs.size());
            _funcs[elem.table_offset + idx++] = i;
        }
    }
}

