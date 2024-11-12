#pragma once

#include "ir.h"
#include "CodeReader.h"
#include "instance.h"

#include <stack>

class Frame;

struct Result {
    enum {
        RETURNED,
        TRHOWN
    }type; 
    union {
        Throwable thrown;
        Value ret;
    }res;
};

class Interpreter {
public:
    Interpreter(
        WasmModule& module,
        Instance& instance,
        bool jit = false):
        _module(module), _instance(instance), _jit(jit){}
    
    Result run(const std::vector<std::string>& mainargs);

private:
    enum {
        RUNNING,
        RETURNING,
        THROWING
    } _state; // Interpreter state
    
    std::vector<Value> _value_stack; // Value stack
    int _params_arity = -1;
    int _return_arity = -1;
    WasmModule& _module;
    Instance& _instance; 
    Frame* _frame = nullptr; 
    bool _jit;


    void _reset(Function* func);
    void _bind(std::vector<Value>& value);
    void _push_frame(Function* func, int arity);
    Frame* _pop_frame();
    Result _run();
    void _exec_op(Opcode_t opcode);
    void _do_return(int fp, SigDecl* sig);
    void _do_func_call(Function* func);
    void _do_jit(Function* func);
    void _advance_caller();
    void* _mem_acc(unsigned size);
    void _clear();
    void _trap();
    void _pop_all_frames();
    Throwable _thrown;
    CodePtr _codeptr;
};
/*
class V3Frame {
	// state for managing the doubly-linked list
	def stack: V3Interpreter;
	def prev: V3Frame;
	def depth: int = if(prev != null, 1 + prev.depth);
	def var next: V3Frame;

	// state for the current activation
	var func: WasmFunction;		// wasm function
	var fp: int;			// frame pointer; i.e. base of locals
	var pc: int;			// program counter
	var stp: int;			// sidetable pointer
	var accessor: V3FrameAccessor;	// accessor, if any

	new(stack, prev) {
		if (prev != null) prev.next = this;
	}
}
*/
class Frame {
protected:
    Frame(Interpreter& interp): interp(interp) {}
    Frame* prev;
    Frame* next;
    Function* func;
    int fp;
    int pc;
    int stp;
    void* accessor;
    Interpreter& interp;
friend class Interpreter;
};