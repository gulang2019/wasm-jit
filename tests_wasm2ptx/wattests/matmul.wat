(module
  (memory 1) ;; Declare a single memory page (64KiB)

  ;; Declare the matmul function
  (func (export "matmul")
    (param $base_a i32)    ;; Base address of matrix A
    (param $base_b i32)    ;; Base address of matrix B
    (param $base_c i32)    ;; Base address of matrix C
    (param $M i32)         ;; Number of rows in A (and C)
    (param $N i32)         ;; Number of columns in B (and C)
    (param $K i32)         ;; Number of columns in A (and rows in B)
    (param $thread_idx i32) ;; Thread index within the block
    (param $block_idx i32)  ;; Block index within the grid
    (param $thread_dim i32) ;; Number of threads per block
    (param $block_dim i32)  ;; Number of blocks in the grid

    ;; Declare locals
    (local $idx i32)       ;; Global thread index
    (local $i i32)         ;; Row index of C
    (local $j i32)         ;; Column index of C
    (local $k i32)         ;; Iteration variable for summation
    (local $addr_a i32)    ;; Address of A[i, k]
    (local $addr_b i32)    ;; Address of B[k, j]
    (local $addr_c i32)    ;; Address of C[i, j]
    (local $sum f64)       ;; Accumulated sum for C[i, j]

    ;; Compute global thread index: blockIdx * threadDim + threadIdx
    local.get $block_idx
    local.get $thread_dim
    i32.mul
    local.get $thread_idx
    i32.add
    local.set $idx

    ;; Bounds check: if idx >= M * N, exit
    local.get $idx
    local.get $M
    local.get $N
    i32.mul
    i32.ge_u
    if (result i32)
      (return)
    end

    ;; Compute row (i) and column (j) indices for C
    local.get $idx
    local.get $N
    i32.div_s
    local.set $i
    local.get $idx
    local.get $N
    i32.rem_s
    local.set $j

    ;; Initialize sum = 0
    f64.const 0
    local.set $sum

    ;; Loop over k (columns of A / rows of B)
    i32.const 0
    local.set $k
    (loop $sum_loop
      ;; If k >= K, exit loop
      local.get $k
      local.get $K
      i32.ge_u
      if (result i32)
        (return)
      end

      ;; Compute addresses for A[i, k] and B[k, j]
      local.get $i
      local.get $K
      i32.mul
      local.get $k
      i32.add
      i32.const 8 ;; sizeof(f64)
      i32.mul
      local.get $base_a
      i32.add
      local.set $addr_a

      local.get $k
      local.get $N
      i32.mul
      local.get $j
      i32.add
      i32.const 8 ;; sizeof(f64)
      i32.mul
      local.get $base_b
      i32.add
      local.set $addr_b

      ;; Load values from A and B, multiply, and add to sum
      local.get $addr_a
      f64.load
      local.get $addr_b
      f64.load
      f64.mul
      local.get $sum
      f64.add
      local.set $sum

      ;; Increment k
      local.get $k
      i32.const 1
      i32.add
      local.set $k

      ;; Repeat loop
      br $sum_loop
    )

    ;; Compute address for C[i, j]
    local.get $i
    local.get $N
    i32.mul
    local.get $j
    i32.add
    i32.const 8 ;; sizeof(f64)
    i32.mul
    local.get $base_c
    i32.add
    local.set $addr_c

    ;; Store sum in C[i, j]
    local.get $addr_c
    local.get $sum
    f64.store
  )
)
