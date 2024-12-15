(module
  (memory 1) ;; Declare a single memory page (64KiB)

  ;; Declare the row_sums function
  (func (export "row_sums")
    (param $base_a i32)      ;; Base address of matrix A
    (param $base_row_sums i32) ;; Base address of rowSums array
    (param $M i32)           ;; Number of rows in A
    (param $N i32)           ;; Number of columns in A
    (param $thread_idx i32)  ;; Thread index within the block
    (param $block_idx i32)   ;; Block index within the grid
    (param $thread_dim i32)  ;; Number of threads per block
    (param $block_dim i32)   ;; Number of blocks in the grid

    ;; Declare locals
    (local $row i32)         ;; Current row index
    (local $col i32)         ;; Current column index
    (local $addr_a i32)      ;; Address of A[row, col]
    (local $addr_row_sum i32) ;; Address of rowSums[row]
    (local $sum f64)         ;; Sum of exponentials
    (local $value f64)       ;; Temporary value for A[row, col]

    ;; Compute global thread index for row
    local.get $block_idx
    local.get $thread_dim
    i32.mul
    local.get $thread_idx
    i32.add
    local.set $row

    ;; Bounds check: if row >= M, exit
    local.get $row
    local.get $M
    i32.ge_u
    if (result i32)
      (return)
    end

    ;; Initialize sum = 0.0
    f64.const 0
    local.set $sum

    ;; Iterate over columns in the current row
    i32.const 0
    local.set $col
    (loop $col_loop
      ;; If col >= N, exit loop
      local.get $col
      local.get $N
      i32.ge_u
      if (result i32)
        (return)
      end

      ;; Compute address of A[row, col]
      local.get $row
      local.get $N
      i32.mul
      local.get $col
      i32.add
      i32.const 8 ;; sizeof(f64)
      i32.mul
      local.get $base_a
      i32.add
      local.set $addr_a

      ;; Load value from A[row, col]
      local.get $addr_a
      f64.load
      local.set $value

      ;; Add exp(value) to sum
      local.get $value
      call $exp ;; Call the imported exp function
      local.get $sum
      f64.add
      local.set $sum

      ;; Increment col
      local.get $col
      i32.const 1
      i32.add
      local.set $col

      ;; Repeat loop
      br $col_loop
    )

    ;; Compute address of rowSums[row]
    local.get $row
    i32.const 8 ;; sizeof(f64)
    i32.mul
    local.get $base_row_sums
    i32.add
    local.set $addr_row_sum

    ;; Store sum in rowSums[row]
    local.get $addr_row_sum
    local.get $sum
    f64.store
  )

  ;; Import the exp function (host-provided)
  (import "env" "exp" (func $exp (param f64) (result f64)))
)
