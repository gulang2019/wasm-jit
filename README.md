## WASM as GPU's Shader Language.

Test cases for WASM to ptx compilation are listed in `tests_wasm2ptx/`, where  
* the wat code are listed in `tests_wasm2ptx/wattests`,
* the cuda code are listed in  `tests_wasm2ptx/cuda`,
* the example ptx codes are listed in  `tests_wasm2ptx/ptx_ground_truth`,

The compiler should output compiled ptx code to `tests_wasm2ptx/ptx` directory.

After that, one can run the benchmark by
```bash 
cd tests_wasm2ptx 
mkdir build && cd build 
cmake .. && make -j32
cd ../.. 
tests_wasm2ptx/build/benchmark [--verbose] TestCase1 TestCase2 ...
```