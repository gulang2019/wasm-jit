#include <cassert>
#include <cmath>
#include <bit>
#include <stdexcept>
#include <csignal>

#include "interpreter.h"
#include "common.h"
#include "utils.h"

Result Interpreter::run(const std::vector<std::string>& mainargs) {
    assert(_module.get_start_fn() == nullptr);
    // evaluate the global variables, treat them as functions
    for (auto& global: _instance._globals){
      if (global._decl.init_expr_bytes.size() > 0) {
        FuncDecl func_decl({
          .sig = new SigDecl({
            .params = {},
            .results = {global._decl.type}
          }),
          .code_bytes = global._decl.init_expr_bytes
        });
        Function func(func_decl, _instance);
        _reset(&func);
        auto res = _run();
        if (res.type == Result::TRHOWN) {
          return res;
        }
        global._value = res.res.ret;
      }
      else {
        global._value = Value(global._decl.type, nullptr);
      }
    }

    // find the main function
    Function* main_fn = nullptr;
    for (auto& export_decl : _module.Exports()) {
        if (export_decl.name == "main" && export_decl.kind == KIND_FUNC) {
            main_fn = &_instance._functions.at(_module.getFuncIdx(export_decl.desc.func));
        }
    }
    assert(main_fn != nullptr);

    // run the main function
    _reset(main_fn);
    std::vector<Value> _args;
    assert(mainargs.size() == main_fn->_decl.sig->params.size());;
    int idx = 0;
    for (auto t: main_fn->_decl.sig->params) {
        _args.push_back(Value(t, mainargs.at(idx)));
        idx++;
    }
    _bind(_args);
    return _run();
}

void Interpreter::_do_jit(Function* func){
  if (func->_decl.sig->results.front() == WASM_TYPE_I32) {
    std::vector<value_t> jit_stack(func->max_stack_offset / sizeof(value_t));
    for (int i = 0; i < _value_stack.size(); i++) {
      jit_stack.at(i) = _value_stack.at(i)._value;
    }
    void* vfp = jit_stack.data();
    void* mem_start = nullptr;
    void* mem_end = nullptr;
    if (_instance._memories.size() > 0) {
      mem_start = _instance._memories.at(0)._data.data();
      mem_end = _instance._memories.at(0)._data.data() + _instance._memories.at(0)._data.size();
    }
    std::vector<value_t> jit_globals;
    for (auto& global: _instance._globals) {
      jit_globals.push_back(global._value._value);
    }
    auto f = func->code_generator->getCode<int (*)(void*, void*, void*, void*)>();
    int res = f(vfp, mem_start, mem_end, jit_globals.data());
    for (size_t i = 0; i < jit_globals.size(); i++) {
      _instance._globals.at(i)._value._value = jit_globals.at(i);
    }
    TRACE("res: %d\n", res);
    _value_stack.push_back(Value(res));
  }
  else if (func->_decl.sig->results.front() == WASM_TYPE_F64) {
    std::vector<value_t> jit_stack(func->max_stack_offset / sizeof(value_t));
    for (int i = 0; i < _value_stack.size(); i++) {
      jit_stack.at(i) = _value_stack.at(i)._value;
    }

    void* vfp = jit_stack.data();
    auto f = func->code_generator->getCode<double (*)(void*)>();
    double res = f(vfp);
    _value_stack.push_back(Value(res));
  }
  else {
    ERR("Invalid return type\n");
  }
}

Result Interpreter::_run(){
/*
if (frame == null || host_outcall != null) return;
			var func = frame.func, pc = frame.pc;
			if (pc == 0) {
				// Entering the function; initialize locals.
				codeptr.iterate_local_codes(pushLocals);
				for (i < func.decl.num_ex_slots) values.push(Values.REF_NULL); // init legacy EH slots
				frame.pc = pc = codeptr.pos; // update pc after decoding locals
				// Fire entry probe(s).
				if (func.decl.entry_probed) {
					var throwable = Instrumentation.fireLocalProbes(DynamicLoc(func, 0, TargetFrame(frame)));
					if (throwable != null) {
						throw(throwable);
						continue;
					}
				}
			}

			// Fire global probes.
			if (Instrumentation.probes != null) {
				var throwable = Instrumentation.fireGlobalProbes(DynamicLoc(func, pc, TargetFrame(frame)));
				if (throwable != null) {
					throw(throwable);
					continue;
				}
			}
			// Read the opcode.
			var b = codeptr.peek1();
			var opcode: Opcode;
			if (b == InternalOpcode.PROBE.code) {
				// First local probes.
				var throwable = Instrumentation.fireLocalProbes(DynamicLoc(func, pc, TargetFrame(frame)));
				if (throwable != null) {
					throw(throwable);
					continue;
				}
				opcode = codeptr.read_orig_opcode(frame.func.decl.orig_bytecode[pc]);
			} else {
				opcode = codeptr.read_opcode();
			}
			execOp(pc, opcode);
*/
    std::signal(SIGFPE, [](int sig) {
      printf("!trap\n");
      exit(-1);
    });
    std::signal(SIGKILL, [](int sig) {
      printf("!trap\n");
      exit(-1);
    });
    if (g_trace) {
          // print the stack
          for (auto& v: _value_stack) {
            v.print(stderr);
          }
        }  
    TRACE("Before Running\n");
    while(_state==RUNNING){
        assert(_frame);
        Function* func = _frame->func;
        if (_frame->pc == 0) {
          int local_cnt = 0;
          for (auto& pure_local: func->_decl.pure_locals){
            for (int i = 0; i < pure_local.count; i++){
              _value_stack.push_back(Value(pure_local.type, nullptr));
            }
            local_cnt += pure_local.count;
          }
          assert(local_cnt == func->_decl.num_pure_locals);
          if (_jit) {
            // Fire entry probe(s).
            _do_jit(func);
            _do_return(_frame->fp, func->_decl.sig);
            continue;
          }
        }
        Opcode_t opcode = _codeptr.rd_opcode();
        // if (g_trace) {
        //   // print the stack
        //   for (auto& v: _value_stack) {
        //     v.print(stderr);
        //   }
        // }
        TRACE("opcode: %s\n", opcode_table[opcode].mnemonic);
        _exec_op(opcode);
    }
    switch (_state) {
      case RETURNING:
        {assert(_return_arity == 1);
        auto ret = _value_stack.back();
        _clear();
        return Result({
          .type = Result::RETURNED,
          .res = {
            .ret = ret
          }
        });}
      case THROWING:
        {return Result({
          .type = Result::TRHOWN,
          .res = {
            .thrown = _thrown
          }
        });}
    };
}

void Interpreter::_clear() {
    _value_stack.clear();
    _params_arity = -1;
    _return_arity = -1;
    _frame = nullptr;
    _state = RUNNING;
}

void Interpreter::_trap() {
  _pop_all_frames();
  _state = THROWING;
}

void Interpreter::_pop_all_frames() {
    while (_frame) {
      _frame = _pop_frame();
    }
}

#define SINGULAR_OP(op, t) { \
    auto a = _value_stack.back(); \
    _value_stack.pop_back(); \
    _value_stack.push_back(Value(op(a._value.t))); \
}

#define BINARY_OP(op, t) { \
    auto b = _value_stack.back(); \
    _value_stack.pop_back(); \
    auto a = _value_stack.back(); \
    _value_stack.pop_back(); \
    _value_stack.push_back(Value(a._value.t op b._value.t)); \
}

#define BINARY_OP_F_2(f, t1, t2) { \
    auto b = _value_stack.back(); \
    _value_stack.pop_back(); \
    auto a = _value_stack.back(); \
    _value_stack.pop_back(); \
    _value_stack.push_back(Value(f(a._value.t1, b._value.t2))); \
}

#define BINARY_OP_F(f, t) BINARY_OP_F_2(f, t, t)

#define LOAD_OP(s_t, r_t){\
  void* ptr = _mem_acc(sizeof(s_t));\
  if (ptr) {_value_stack.push_back(Value((r_t)(*(s_t*)(ptr))));}\
}\

#define STORE_OP(s_t, r_t) {\
  auto v = _value_stack.back();\
  _value_stack.pop_back();\
  void*ptr = _mem_acc(sizeof(r_t));\
  if (ptr) {*(r_t*)ptr = r_t(v._value.s_t);}\
}\

#define JUMP() {\
  _codeptr.at(_frame->func->_j_table.at(_codeptr.offset()));\
}\


// #define BINARY_OP_S(op) { \
//     auto b = _value_stack.back(); \
//     _value_stack.pop_back(); \
//     auto a = _value_stack.back(); \
//     _value_stack.pop_back(); \
//     _value_stack.push_back(Value(a._value.i32 op b._value.i32)); \
// }

void Interpreter::_exec_op(Opcode_t opcode) {
    switch (opcode)
    {
    case WASM_OP_UNREACHABLE:		{
        _thrown = Throwable({
          .reason = "unreachable",
          .file = __FILE__,
          .lineno = __LINE__
        });
        _trap();
        break;
  }
  case WASM_OP_NOP:			{
    break;
  }
  case WASM_OP_BLOCK:		
  case WASM_OP_LOOP:		{
    _codeptr.skip_block_type();
    break;
  }
  case WASM_OP_IF:			{
    auto v = _value_stack.back();
    _value_stack.pop_back();
    assert(v._type == WASM_TYPE_I32);
    _codeptr.skip_block_type();
    if (!v._value.i32) {
      JUMP();
    }
    break;
  }
  case WASM_OP_ELSE:		{
    JUMP();
    break;
  }
  case WASM_OP_END:			{
    if (_codeptr.is_end()) {
        _do_return(_frame->fp, _frame->func->_decl.sig);
    }
    break;
  }
  case WASM_OP_BR:			{
    auto index = _codeptr.rd_u32leb();
    JUMP();
    break;
  }
  case WASM_OP_BR_IF:		{
    auto index = _codeptr.rd_u32leb();
    auto v = _value_stack.back();
    _value_stack.pop_back();
    assert(v._type == WASM_TYPE_I32);
    if (v._value.i32) {
      JUMP();
    }
    break;
  }
  case WASM_OP_BR_TABLE:		{
    ERR("WASM_OP_BR_TABLE not implemented");
    break;
  }
  case WASM_OP_RETURN:		{
    _do_return(_frame->fp, _frame->func->_decl.sig);
    break;
  }
  case WASM_OP_CALL:		{
    auto func_index = _codeptr.rd_u32leb();
    _do_func_call(&_instance._functions[func_index]);
    break;
  }
  case WASM_OP_CALL_INDIRECT:	{
    auto sig_index = _codeptr.rd_u32leb();
    auto table_index = _codeptr.rd_u32leb();
    auto func_index = _value_stack.back()._value.u32;
    _value_stack.pop_back();
    auto sig = _module.getSig(sig_index);
    auto tab = _instance._tables.at(table_index);
    TRACE("tab.limits.initial: %d\n", tab._decl.limits.initial);
    TRACE("tab.limits.max: %d\n", tab._decl.limits.max);
    TRACE("tab.limits.has_max: %d\n", tab._decl.limits.has_max);
    TRACE("tab.flags: %d\n", tab._decl.limits.flag);
    if (func_index >= tab._decl.limits.initial) {
      _thrown = Throwable({
        .reason = "function index out of bounds",
        .file = __FILE__,
        .lineno = __LINE__
      });
      _trap();
    }
    else if (tab._funcs.at(func_index) == nullptr) {
      _thrown = Throwable({
        .reason = "uninitialized element",
        .file = __FILE__,
        .lineno = __LINE__
      });
      _trap();
    }
    else if (tab._funcs.at(func_index)->sig != sig) {
      _thrown = Throwable({
        .reason = "function signature mismatch",
        .file = __FILE__,
        .lineno = __LINE__
      });
      _trap();
    }
    else 
      _do_func_call(&_instance._functions.at(_module.getFuncIdx(tab._funcs.at(func_index))));
    break;
  }
  case WASM_OP_DROP:		{
    _value_stack.pop_back();
    break;
  }
  case WASM_OP_SELECT:		{
    ERR("WASM_OP_SELECT not implemented");
    break;
  }
  case WASM_OP_LOCAL_GET:		{
    _value_stack.push_back(_value_stack[_frame->fp + _codeptr.rd_u32leb()]);
    break;
  }
  case WASM_OP_LOCAL_SET:		{
    auto v = _value_stack.back();
    _value_stack.pop_back();
    _value_stack[_frame->fp + _codeptr.rd_u32leb()] = v;
    break;
  }
  case WASM_OP_LOCAL_TEE:		{
    _value_stack[_frame->fp + _codeptr.rd_u32leb()] = _value_stack.back();
    break;
  }
  case WASM_OP_GLOBAL_GET:		{
    auto idx = _codeptr.rd_u32leb();
    _value_stack.push_back(_instance._globals.at(idx)._value);
    break;
  }
  case WASM_OP_GLOBAL_SET:		{
    auto idx = _codeptr.rd_u32leb();
    assert (_instance._globals.at(idx)._decl.is_mutable);
    auto v = _value_stack.back();
    _value_stack.pop_back();
    _instance._globals.at(idx)._value = v;
    break;
  }
  case WASM_OP_I32_LOAD:		{
    LOAD_OP(int, int);
    break;
  }
  case WASM_OP_F64_LOAD:		{
    LOAD_OP(double, double);
    break;
  }
  case WASM_OP_I32_LOAD8_S:		{
    LOAD_OP(int8_t, int);
    break;
  }
  case WASM_OP_I32_LOAD8_U:		{
    LOAD_OP(uint8_t, unsigned int);
    break;
  }
  case WASM_OP_I32_LOAD16_S:	{
    LOAD_OP(short, int);
    break;
  }
  case WASM_OP_I32_LOAD16_U:	{
    LOAD_OP(unsigned short, unsigned int);
    break;
  }
  case WASM_OP_I32_STORE:		{
    STORE_OP(i32, int);
    break;
  }
  case WASM_OP_F64_STORE:		{
    STORE_OP(f64, double);
    break;
  }
  case WASM_OP_I32_STORE8:		{
    STORE_OP(i32, int8_t);
    break;
  }
  case WASM_OP_I32_STORE16:		{
    STORE_OP(i32, short);
    break;
  }
  case WASM_OP_MEMORY_SIZE:		{
    auto index = _codeptr.rd_u32leb();
    _value_stack.push_back(Value(_instance._memories.at(index).size()));
    break;
  }
  case WASM_OP_MEMORY_GROW:		{
    auto index = _codeptr.rd_u32leb();
    auto delta = _value_stack.back()._value.i32;
    _value_stack.push_back(Value(_instance._memories.at(index).grow(delta)));
    break;
  }
  case WASM_OP_I32_CONST:		{
    _value_stack.push_back(Value(_codeptr.rd_i32leb()));
    break;
  }
  case WASM_OP_F64_CONST:		{
    auto v = _codeptr.rd_u64();
    _value_stack.push_back(Value(WASM_TYPE_F64, &v));
    break;
  }
  case WASM_OP_I32_EQZ:		{
    SINGULAR_OP(0 ==, i32);
    break;
  }
  case WASM_OP_I32_EQ:		{
    BINARY_OP(==, i32);
    break;
  }
  case WASM_OP_I32_NE:		{
    BINARY_OP(!=, i32);
    break;
  }
  case WASM_OP_I32_LT_S:		{
    BINARY_OP(<, i32);
    break;
  }
  case WASM_OP_I32_LT_U:		{
    BINARY_OP(<, u32);
    break;
  }
  case WASM_OP_I32_GT_S:		{
    BINARY_OP(>, i32);
    break;
  }
  case WASM_OP_I32_GT_U:		{
    BINARY_OP(>, u32);
    break;
  }
  case WASM_OP_I32_LE_S:		{
    BINARY_OP(<=, i32);
    break;
  }
  case WASM_OP_I32_LE_U:		{
    BINARY_OP(<=, u32);
    break;
  }
  case WASM_OP_I32_GE_S:		{
    BINARY_OP(>=, i32);
    break;
  }
  case WASM_OP_I32_GE_U:		{
    BINARY_OP(>=, u32);
    break;
  }
  case WASM_OP_F64_EQ:		{
    BINARY_OP(==, f64);
    break;
  }
  case WASM_OP_F64_NE:		{
    BINARY_OP(!=, f64);
    break;
  }
  case WASM_OP_F64_LT:		{
    BINARY_OP(<, f64);
    break;
  }
  case WASM_OP_F64_GT:		{
    BINARY_OP(>, f64);
    break;
  }
  case WASM_OP_F64_LE:		{
    BINARY_OP(<=, f64);
    break;
  }
  case WASM_OP_F64_GE:		{
    BINARY_OP(>=, f64);
    break;
  }
  case WASM_OP_I32_CLZ:		{
    SINGULAR_OP(__builtin_clz, i32);
    break;
  }
  case WASM_OP_I32_CTZ:		{
    SINGULAR_OP(__builtin_ctz, i32);
    break;
  }
  case WASM_OP_I32_POPCNT:		{
    SINGULAR_OP(std::popcount, u32);
    break;
  }
  case WASM_OP_I32_ADD:		{
    BINARY_OP(+, i32);
    break;
  }
  case WASM_OP_I32_SUB:{
    BINARY_OP(-, i32);
    break;
  }
  case WASM_OP_I32_MUL:		{
    BINARY_OP(*, i32);
    break;
  }
  case WASM_OP_I32_DIV_S:		{
    BINARY_OP(/, i32);
    break;
  }
  case WASM_OP_I32_DIV_U:	{
    BINARY_OP(/, u32);	
    break;
  }
  case WASM_OP_I32_REM_S:		{
    BINARY_OP(/, i32);
    break;
  }
  case WASM_OP_I32_REM_U:		{
    BINARY_OP(/, u32);
    break;
  }
  case WASM_OP_I32_AND:		{
    BINARY_OP(&, i32);
    break;
  }
  case WASM_OP_I32_OR:		{
    BINARY_OP(|, i32);
    break;
  }
  case WASM_OP_I32_XOR:		{
    BINARY_OP(^, i32);
    break;
  }
  case WASM_OP_I32_SHL:		{
    BINARY_OP(<<, i32);
    break;
  }
  case WASM_OP_I32_SHR_S:		{
    BINARY_OP(>>, i32);
    break;
  }
  case WASM_OP_I32_SHR_U:		{
    BINARY_OP(>>, u32);
    break;
  }
  case WASM_OP_I32_ROTL:		{
    BINARY_OP_F_2(std::rotl, u32, i32);
    break;
  }
  case WASM_OP_I32_ROTR:		{
    BINARY_OP_F_2(std::rotr, u32, i32);
    break;
  }
  case WASM_OP_F64_ABS:		{
    SINGULAR_OP(std::abs, f64);
    break;
  }
  case WASM_OP_F64_NEG:		{
    SINGULAR_OP(-, f64);
    break;
  }
  case WASM_OP_F64_CEIL:		{
    SINGULAR_OP(std::ceil, f64);
    break;
  }
  case WASM_OP_F64_FLOOR:		{
    SINGULAR_OP(std::floor, f64);
    break;
  }
  case WASM_OP_F64_TRUNC:		{
    SINGULAR_OP(std::trunc, f64);
    break;
  }
  case WASM_OP_F64_NEAREST:		{
    SINGULAR_OP(std::round, f64);
    break;
  }
  case WASM_OP_F64_SQRT:		{
    SINGULAR_OP(std::sqrt, f64);
    break;
  }
  case WASM_OP_F64_ADD:		{
    BINARY_OP(+, f64);
    break;
  }
  case WASM_OP_F64_SUB:		{
    BINARY_OP(-, f64);
    break;
  }
  case WASM_OP_F64_MUL:		{
    BINARY_OP(*, f64);
    break;
  }
  case WASM_OP_F64_DIV:		{
    auto b = _value_stack.back();
    _value_stack.pop_back();
    auto a = _value_stack.back();
    _value_stack.pop_back();
    if (b._value.f64 == 0.0) {
      _thrown = Throwable({
        .reason = "division by zero",
        .file = __FILE__,
        .lineno = __LINE__
      });
      _trap();
    }
    else {
      _value_stack.push_back(Value(a._value.f64 / b._value.f64));
    }
    break;
  }
  case WASM_OP_F64_MIN:		{
    BINARY_OP_F(std::min, f64);
    break;
  }
  case WASM_OP_F64_MAX:		{
    BINARY_OP_F(std::max, f64);
    break;
  }
  case WASM_OP_I32_TRUNC_F64_S:	{
    ERR("WASM_OP_I32_TRUNC_F64_S not implemented");
    break;
  }
  case WASM_OP_I32_TRUNC_F64_U:	{
    ERR("WASM_OP_I32_TRUNC_F64_U not implemented");
    break;
  }
  case WASM_OP_F64_CONVERT_I32_S:	{
    ERR("WASM_OP_F64_CONVERT_I32_S not implemented");
    break;
  }
  case WASM_OP_F64_CONVERT_I32_U:	{
    ERR("WASM_OP_F64_CONVERT_I32_U not implemented");
    break;
  }
  case WASM_OP_I32_EXTEND8_S:	{
    ERR("WASM_OP_I32_EXTEND8_S not implemented");
    break;
  }
  case WASM_OP_I32_EXTEND16_S:	{
    ERR("WASM_OP_I32_EXTEND16_S not implemented");
    break;
  }
    default: {
        ERR("opcode not implemented");
        break;
    }
  };
  if (_frame) _frame->pc = _codeptr.offset();
}

void Interpreter::_do_func_call(Function* func) {
    if (_frame) _codeptr.at(_frame->pc);
    _push_frame(func, func->_decl.sig->params.size());
}

void Interpreter::_do_return(int fp, SigDecl* sig) {
    auto count = sig->results.size();
    _frame = _pop_frame();
    if (_frame == nullptr) {
        assert(_return_arity == count);
        _state = RETURNING;
    } else {
      _advance_caller();
    }
}

void* Interpreter::_mem_acc(unsigned size) {
  assert(_value_stack.back()._type == WASM_TYPE_I32);
  auto offset = _value_stack.back()._value.i32;
  _value_stack.pop_back();
  auto mem_arg = _codeptr.rd_mem_arg();
  auto& memory = _instance._memories.at(mem_arg.memory_index);
  void* ptr = memory.at(offset + mem_arg.offset, size, &_thrown);
  if (!ptr)
    _trap();
  return ptr;
}

void Interpreter::_advance_caller() {
    auto op_code = _codeptr.rd_opcode();
    switch (op_code)
    {
        case WASM_OP_CALL:{
            _codeptr.skip_leb();
            _frame->pc = _codeptr.offset();
            break;
        }
        default:
            ERR("Adavance Caller not implemented");
    };
}

void Interpreter::_reset(Function* func) {
    _push_frame(func, 0);
    _params_arity = func->_decl.sig->params.size();
    _return_arity = func->_decl.sig->results.size();
    _state = RUNNING;
}

void Interpreter::_bind(std::vector<Value>& value) {
    assert(_params_arity == value.size());;
    _params_arity -= value.size();
    for (auto& v : value) {
        _value_stack.push_back(v);
    }
}

void Interpreter::_push_frame(Function* func, int arity) {
    /*
    wf: WasmFunction => {
        var next = frame = nextFrame(frame);
        if (next.depth > Execute.limits.max_call_depth) {
            trap(TrapReason.STACK_OVERFLOW);
            return;
        }
        next.func = wf;
        next.fp = values.top - arity;
        next.pc = 0;
        next.stp = 0;
        next.accessor = null;
        var code = wf.decl.cur_bytecode;
        codeptr.reset(code, 0, code.length);
    }
    */
    Frame* next = new Frame(*this);
    next->prev = _frame;
    next->next = nullptr;
    if (_frame) {
      _frame -> next = next;
    }
    _frame = next;
    next->func = func;
    next->fp = _value_stack.size() - arity;
    next->pc = 0;
    next->stp = 0;
    next->accessor = nullptr;
    _codeptr.reset(&func->_decl.code_bytes, 0, func->_decl.code_bytes.size());
    // var code = wf.decl.cur_bytecode;
    // codeptr.reset(code, 0, code.length);
}

/*
def nextFrame(caller: V3Frame) -> V3Frame {
		if (caller == null) {
			if (cache == null) return cache = V3Frame.new(this, null);
			return cache;
		}
		var next = caller.next;
		if (next == null) next = caller.next = V3Frame.new(this, caller);
		return next;
	}
*/
// Frame* Interpreter::_next_frame(Frame* caller){
//     // if (caller == nullptr) {
//     //     if (_frame == nullptr) {
          
//     //        _frame = new Frame(*this);
//     //     }
//     //     return _frame;
//     // }
//     // Frame* next = caller->next;
//     // if (next == nullptr) next = caller->next = new Frame(*this);

//     // return next;
// }

Frame* Interpreter::_pop_frame(){
    if (_frame == nullptr) return nullptr;
    _frame->accessor = nullptr;
    auto __frame = _frame;
    _frame = _frame->prev;
    delete __frame;
    if (_frame == nullptr) {
        _codeptr.reset(nullptr, 0, 0);
        return nullptr;
    } else {
        _frame->next = nullptr;
        _codeptr.reset(&_frame->func->_decl.code_bytes, _frame->pc, _frame->func->_decl.code_bytes.size());
        return _frame;
    }
}