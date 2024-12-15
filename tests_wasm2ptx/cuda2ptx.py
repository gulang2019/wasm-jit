import os 

def compile(dirname: str, output_dir: str = None):
    # walk the directory and compile all .cu files to .ptx
    
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(".cu"):
                print(f"Compiling {file}")
                output_file = os.path.join(output_dir, file.replace('.cu', '.ptx'))
                
                os.system(f"nvcc -ptx {os.path.join(root, file)} -g -lineinfo --source-in-ptx -keep -o {output_file}")
                
if __name__ == "__main__":
    compile("tests_wasm2ptx/cuda", "tests_wasm2ptx/ptx")
    