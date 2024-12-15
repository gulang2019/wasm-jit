(module
  (memory 1) ;; Declare a single memory page (64KiB)

  ;; Declare the softmax function
  (func (export "softmax")
    (param $base_a i32)        ;; Base address of matrix A
    (param $base_row_sums i32) ;; Base address of rowSums array
    (param $base_b i32)        ;; Base address of matrix B
    (param $M i32)             ;; Number of rows in A (and B)
    (param $N i32)             ;; Number of columns in A (and B)
    (param $thread_idx i32)    ;; Thread index within the block
    (param $block_idx i32)     ;; Block index within the grid
    (param $thread_dim i32)    ;; Number of threads per block
    (param $block_dim i32)     ;; Number of blocks in the grid

    ;; Declare locals
    (local $idx i32)           ;; Global thread index
    (local $total i32)         ;; Total number of elements (M * N)
    (local $row i32)           ;; Current row index
    (local $col i32)           ;; Current column index
    (local $addr_a i32)        ;; Address of A[idx]
    (local $addr_row_sum i32)  ;; Address of rowSums[row]
    (local $addr_b i32)        ;; Address of B[idx]
    (local $value f64)         ;; Value from A[idx]
    (local $row_sum f64)       ;; Value from rowSums[row]
    (local $result f64)        ;; Result of softmax computation

    ;; Compute global thread index
    local.get $block_idx
    local.get $thread_dim
    i32.mul
    local.get $thread_idx
    i32.add
    local.set $idx

    ;; Compute total number of elements (M * N)
    local.get $M
    local.get $N
    i32.mul
    local.set $total

    ;; Bounds check: if idx >= total, exit
    local.get $idx
    local.get $total
    i32.ge_u
    if (result i32)
      (return)
    end

    ;; Compute row (i) and column (j) indices
    local.get $idx
    local.get $N
    i32.div_s
    local.set $row
    local.get $idx
    local.get $N
    i32.rem_s
    local.set $col

    ;; Compute address of A[idx]
    local.get $idx
    i32.const 8 ;; sizeof(f64)
    i32.mul
    local.get $base_a
    i32.add
    local.set $addr_a

    ;; Compute address of rowSums[row]
    local.get $row
    i32.const 8 ;; sizeof(f64)
    i32.mul
    local.get $base_row_sums
    i32.add
    local.set $addr_row_sum

    ;; Compute address of B[idx]
    local.get $idx
    i32.const 8 ;; sizeof(f64)
    i32.mul
    local.get $base_b
    i32.add
    local.set $addr_b

    ;; Load value from A[idx]
    local.get $addr_a
    f64.load
    local.set $value

    ;; Load value from rowSums[row]
    local.get $addr_row_sum
    f64.load
    local.set $row_sum

    ;; Compute exp(A[idx]) / rowSums[row]
    local.get $value
    call $exp ;; Call the imported exp function
    local.get $row_sum
    f64.div
    local.set $result

    ;; Store result in B[idx]
    local.get $addr_b
    local.get $result
    f64.store
  )
)
