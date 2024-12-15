#include "ptx_compiler.h"

#include "CodeReader.h"

void PTXCompiler::compile(const FuncDecl *func) {
    auto codeptr = CodePtr();
    codeptr.reset(&func->code_bytes, 0, func->code_bytes.size());
    for (auto p: func->sig->params) stack.push(p);
}
