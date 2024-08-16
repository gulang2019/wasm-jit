#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE}))
TEST_DIR=$SCRIPT_DIR/tests
WASM_VM_PATH=${1:-$SCRIPT_DIR/wasm-vm}

# Directories
wasm_dir="$TEST_DIR/wasm"
expect_dir="$TEST_DIR/expect"


# Function to run a wasm file with wasmtime and get stdout and stderr
function run_wasm {
    local wasm_file="$1"
    shift
    local run_args=$@
    if [ -z "$run_args" ]; then
       local output=$(timeout 5 $WASM_VM_PATH --jit "$wasm_file" 2>&1)
     else
       local output=$(timeout 5 $WASM_VM_PATH --jit -a $run_args "$wasm_file" 2>&1)
     fi
    if [ $? -ne 0 ]; then
      echo "timeout"
    else
      echo "$output"
    fi
}

function check_output {
    local wasm_file="$1"
    shift
    local expected_output="$1"
    shift
    local run_args="$@"

    local actual_output=$(run_wasm "$wasm_file" $run_args)

    local wasm_filename=$(basename "$wasm_file" .wasm)
    local run_args_print=${run_args:-(void)}

    if [ "$actual_output" == "$expected_output" ]; then
        echo "$wasm_filename (run_args: ${run_args_print}) ... success!"
        return 0
    else
        echo "$wasm_filename (run_args: ${run_args_print}) ... fail!"
        echo "  Expected output:"
        echo "  -------------------"
        echo "$expected_output"
        echo "  -------------------"
        echo "  Actual output:"
        echo "  -------------------"
        echo "$actual_output"
        echo "  -------------------"
        return 1
    fi
}

num_tests=0
pass=0
# Iterate over all .wasm files in the wasm directory
for wasm_file in "$wasm_dir"/*.wasm; do
    wasm_filename=$(basename "$wasm_file" .wasm)
    
    # Check for corresponding .expect file
    expect_file="$expect_dir/$wasm_filename.expect"
    # Check for corresponding .runs file
    runs_file="$expect_dir/$wasm_filename.runs"
    
    echo "#### Test $((num_tests+1)) ####"
    if [ -f "$expect_file" ]; then
        expected_output=$(cat "$expect_file")
        check_output $wasm_file "$expected_output"
        if [ $? -eq 0 ]; then
          ((pass++))
        fi
        ((num_tests++))
    elif [ -f "$runs_file" ]; then
        runs_fail=0
        while IFS= read -r line; do
            if [[ "$line" =~ (.*)=(.*) ]]; then
                run_args=${BASH_REMATCH[1]}
                expected_output="${BASH_REMATCH[2]}"
                check_output $wasm_file $expected_output $run_args
                if [ $? -ne 0 ]; then
                  ((runs_fail++))
                fi
            fi
        done < "$runs_file"
        if [ $runs_fail -eq 0 ]; then
          ((pass++))
        fi
        ((num_tests++))
    else
        echo "$wasm_filename... no .expect or .runs file found!"
    fi
    echo ""
done

fail=$((num_tests-pass))
prop10=$(echo "scale=3; ($pass/$num_tests)*10" | bc)
echo ""
echo "==================== SUMMARY ===================="
printf '# Tests: %d | Pass: %d; Fail: %d | Score: %f\n' $num_tests $pass $fail $prop10
echo "================================================="
exit $fail
