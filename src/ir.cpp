#include "ir.h"

#include <chrono>

const char* wasm_type_string(wasm_type_t type) {
  switch (type) {
    case WASM_TYPE_I32:       return "i32"; break;
    case WASM_TYPE_I64:       return "i64"; break;
    case WASM_TYPE_F32:       return "f32"; break;
    case WASM_TYPE_F64:       return "f64"; break;
    case WASM_TYPE_V128:      return "v128"; break;
    case WASM_TYPE_EXTERNREF: return "externref"; break;
    case WASM_TYPE_FUNCREF:   return "funcref"; break;
    default:  
      return "unknown";
  }
}

const char* wasm_kind_string(wasm_kind_t kind) {
  switch (kind) {
    case KIND_FUNC:     return "func"; break;
    case KIND_TABLE:    return "table"; break;
    case KIND_MEMORY:   return "memory"; break;
    case KIND_GLOBAL:   return "global"; break;
    default:  
      return "unknown";
  }
}

// Returns the string name of a section code.
const char* wasm_section_name(byte code) {
  switch (code) {
  case WASM_SECT_TYPE: return "type";
  case WASM_SECT_IMPORT: return "import";
  case WASM_SECT_FUNCTION: return "function";
  case WASM_SECT_TABLE: return "table";
  case WASM_SECT_MEMORY: return "memory";
  case WASM_SECT_GLOBAL: return "global";
  case WASM_SECT_EXPORT: return "export";
  case WASM_SECT_START: return "start";
  case WASM_SECT_ELEMENT: return "element";
  case WASM_SECT_CODE: return "code";
  case WASM_SECT_DATA: return "data";
  case WASM_SECT_DATACOUNT: return "datacount";
  case WASM_SECT_CUSTOM: return "custom";
  default:
    return "unknown";
  }
}


#define CACHE 1

#if CACHE == 0
#define REASSIGN(val, ispace, rettype) ({ \
  this->get##ispace(mod.get##ispace##Idx(val)); \
})
#else
#define REASSIGN(val, ispace, rettype) ({ \
  auto &cached_val = reassign_cache[val]; \
  if (!cached_val) {  \
    cached_val = this->get##ispace(mod.get##ispace##Idx(val));  \
  } \
  (rettype*) cached_val; \
})
#endif


WasmModule& WasmModule::deepcopy(const WasmModule &mod, const char* log_str) {
  std::unordered_map<void*, void*> reassign_cache;

  this->magic = mod.magic;
  this->version = mod.version;

  //
  this->customs = mod.customs;
  this->sigs = mod.sigs;

  this->imports = mod.imports;

  this->funcs = mod.funcs;
  this->tables = mod.tables;
  this->mems = mod.mems;
  this->globals = mod.globals;

  this->exports = mod.exports;
  
  this->elems = mod.elems;

  this->datas = mod.datas;

  this->start_fn = mod.start_fn ? REASSIGN(mod.start_fn, Func, FuncDecl) : NULL;

  this->has_datacount = mod.has_datacount;
  this->num_datas_datacount = mod.num_datas_datacount;

  // Patching
  DescriptorPatch<ImportDecl> (this->imports.list, mod, reassign_cache);
  DescriptorPatch<ExportDecl> (this->exports, mod, reassign_cache);
  FunctionPatch (mod, reassign_cache);
  // Element Patch
  for (auto &elem : this->elems) {
    for (auto &fn_ptr : elem.func_indices) {
      fn_ptr = REASSIGN(fn_ptr, Func, FuncDecl);
    }
  }
  // Data Patch
  for (auto &data : this->datas) {
    if (data.flag != 1) {
      data.mem = REASSIGN(data.mem, Memory, MemoryDecl);
    }
  }

  CustomPatch (mod, reassign_cache);
  return *this;
}



/* Descriptor patching for copy constructor */
template<typename T>
void WasmModule::DescriptorPatch (std::list<T> &list, const WasmModule &mod, std::unordered_map<void*, void*> &reassign_cache) {
  for (auto &v : list) {
    switch (v.kind) {
      case KIND_FUNC: v.desc.func = REASSIGN(v.desc.func, Func, FuncDecl); break;
      case KIND_TABLE: v.desc.table = REASSIGN(v.desc.table, Table, TableDecl); break;
      case KIND_MEMORY: v.desc.mem = REASSIGN(v.desc.mem, Memory, MemoryDecl); break;
      case KIND_GLOBAL: v.desc.global = REASSIGN(v.desc.global, Global, GlobalDecl); break;
      default:
        ERR("Import/Export kind: %u\n", v.kind);
        throw std::runtime_error("Invalid import/export kind");
    }
  }
}


void WasmModule::FunctionPatch (const WasmModule &mod, std::unordered_map<void*, void*> &reassign_cache) {
  for (auto &func : this->funcs) {
    func.sig = REASSIGN(func.sig, Sig, SigDecl);
  }
}


void WasmModule::CustomPatch (const WasmModule &mod, std::unordered_map<void*, void*> &reassign_cache) {
  for (auto &custom : this->customs) {
    if (custom.name == "name") {
      DebugNameDecl &debug = custom.debug;
      if (!debug.func_assoc.empty()) {
        for (auto &name_asc : debug.func_assoc) {
          name_asc.func = REASSIGN(name_asc.func, Func, FuncDecl);
        }
      }
      this->fn_names_debug = &debug.func_assoc;
    }
  }
}


WasmModule& WasmModule::operator= (const WasmModule &mod) {
  return this->deepcopy(mod, "Assign Module");
}

WasmModule::WasmModule (const WasmModule &mod) {
  this->deepcopy(mod, "Copy Module");
}
