(module
  (memory 1) ;; Declare a single page of memory (64KiB)

  ;; Declare the relu function
  (func (export "relu")
    (param $base_a i32)    ;; Base address of array A
    (param $base_c i32)    ;; Base address of array C
    (param $size i32)      ;; Total number of elements
    (param $thread_idx i32) ;; Thread index within the block
    (param $block_idx i32)  ;; Block index within the grid
    (param $thread_dim i32) ;; Number of threads per block
    (param $block_dim i32)  ;; Number of blocks in the grid

    ;; Declare locals
    (local $global_idx i32) ;; Global thread index
    (local $addr_a i32)     ;; Address of A[global_idx]
    (local $addr_c i32)     ;; Address of C[global_idx]
    (local $value i32)      ;; Value from A[global_idx]

    ;; Compute global thread index: blockIdx * threadDim + threadIdx
    local.get $block_idx
    local.get $thread_dim
    i32.mul
    local.get $thread_idx
    i32.add
    local.set $global_idx

    ;; Bounds check: if global_idx >= size, exit
    local.get $global_idx
    local.get $size
    i32.ge_u
    if (result i32)
      (return)
    end

    ;; Compute addresses for A[global_idx] and C[global_idx]
    local.get $global_idx
    i32.const 4
    i32.mul
    local.get $base_a
    i32.add
    local.set $addr_a

    local.get $global_idx
    i32.const 4
    i32.mul
    local.get $base_c
    i32.add
    local.set $addr_c

    ;; Load value from A[global_idx]
    local.get $addr_a
    i32.load
    local.set $value

    ;; Compute max(0, value)
    local.get $value
    i32.const 0
    i32.gt_s
    if
      ;; If value > 0, store value in C[global_idx]
      local.get $value
    else
      ;; Otherwise, store 0 in C[global_idx]
      i32.const 0
    end

    ;; Store the result in C[global_idx]
    local.get $addr_c
    i32.store
  )
)
