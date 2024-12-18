#include <iostream>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <cassert>
#include <fstream>
#include <chrono>

#include "common.h"
#include "parse.h"
#include "ir.h"
#include "interpreter.h"
#include "instance.h"
#include "ptx_compiler.h"


static struct option long_options[] = {
  {"trace", no_argument,  &g_trace, 1},
  {"args", optional_argument, NULL, 'a'},
  {"help", no_argument, NULL, 'h'},
  {"jit", no_argument, NULL, 'j'}
};

typedef struct args_t {
  std::string infile;
  const char* outfile;
  std::vector<std::string> mainargs;
  bool jit = false; 
} args_t;

args_t parse_args(int argc, char* argv[]) {
  int opt;
  args_t args;
  optind = 0;
  while ((opt = getopt_long_only(argc, argv, ":a:o:hj", long_options, NULL)) != -1) {
    switch(opt) {
      case 0: break;
      case 'a':
        args.mainargs.push_back(optarg);
        while ((optind < (argc - 1))) {
          args.mainargs.push_back(argv[optind++]);
        }
        break;
      case 'o':
        if (optarg == NULL) {
          ERR("Option '-o' requires an argument.\n");
          exit(1);
        }
        args.outfile = optarg;
        break;
      case 'j':
        args.jit = true;
        break;
      case 'h':
      default:
        ERR("Usage: %s [--trace (optional)]  [-a <space-separated args>] <input-file>\n", argv[0]);
        exit(opt != 'h');
    }
  }

  if ( ((optind + 1) != argc)) {
    ERR("Executable takes one positional argument \"filename\"\n");
    exit(1);
  }

  args.infile = std::string(argv[optind]);
  return args;
}

// Main function.
// Parses arguments and either runs a file with arguments.
//  --trace: enable tracing to stderr
int main(int argc, char *argv[]) {
  args_t args = parse_args(argc, argv);
    
  byte* start = NULL;
  byte* end = NULL;

  const char *infile = args.infile.c_str();
  ssize_t r = load_file(infile, &start, &end);
  if (r < 0) {
    ERR("failed to load: %s\n", infile);
    return 1;
  }

  TRACE("loaded %s: %ld bytes\n", infile, r);
  WasmModule module = parse_bytecode(start, end);
  unload_file(&start, &end);
  
  /* */
  /* Interpreter here */
  /* */
  Instance instance(module);
  // Interpreter interp(module, instance, args.jit);
  // Result res = interp.run(args.mainargs);
  // if (res.type == Result::TRHOWN) {
  //   // ERR("%s, %d: %s\n", \
  //   //     res.res.thrown.file, \
  //   //     res.res.thrown.lineno\
  //   //     , res.res.thrown.reason);
  //   printf("!trap\n");
  //   return 0;
  // }
  // else if (res.type == Result::RETURNED) {
  //   auto& v = res.res.ret;
  //   v.print();
  // }

  Function* fn = nullptr;
  
  int fn_idx = -1;
  const char* fn_name = nullptr;
  for (auto &[name, kind, desc] : module.Exports()) {
    if (kind == KIND_FUNC) {
      fn_idx = module.getFuncIdx(desc.func);
      fn = &instance._functions.at(fn_idx);
      fn_name = name.c_str();
      break;
    }
  }
  assert(fn != nullptr && fn_name);

  PTXCompiler cc;
  TRACE("PTX JIT on Function #%s\n", fn_name);
  auto start_time = std::chrono::high_resolution_clock::now();
  cc.compile(fn_name, fn->_decl);
  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time in milliseconds
  std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

  // Output the result
  std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

  if (args.outfile) {
    std::ofstream out(args.outfile);
    cc.emit(out);
  } 
  // cc.emit(std::cout);

  return 0;
}




