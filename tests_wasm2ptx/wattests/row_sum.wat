(module
  (memory 1) ;; Declare a single memory page (64KiB)

  ;; Declare the row_sums function
  (func (export "row_sum-ppii")
    (param $thread_idx i32)  ;; Thread index within the block
    (param $block_idx i32)   ;; Block index within the grid
    (param $thread_dim i32)  ;; Number of threads per block
    (param $base_a i32)      ;; Base address of matrix A
    (param $base_row_sums i32) ;; Base address of rowSums array
    (param $M i32)           ;; Number of rows in A
    (param $N i32)           ;; Number of columns in A

    ;; Declare locals
    (local $row i32)         ;; Current row index
    (local $col i32)         ;; Current column index
    (local $sum f64)         ;; Sum of exponentials

    ;; Compute global thread index for row
    local.get $block_idx
    local.get $thread_dim
    i32.mul
    local.get $thread_idx
    i32.add
    local.set $row

    ;; Initialize sum = 0.0
    f64.const 0
    local.set $sum

    ;; Iterate over columns in the current row
    i32.const 0
    local.set $col
    (loop $col_loop
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

      ;; Load value from A[row, col]
      f64.load
      local.get $sum
      f64.add
      local.set $sum

      ;; Increment col
      local.get $col
      i32.const 1
      i32.add
      local.set $col

      ;; Repeat loop
      local.get $col
      local.get $N
      i32.lt_s
      br_if $col_loop ;; bra $L__BB0_5;
    )

    ;; Compute address of rowSums[row]
    local.get $row
    i32.const 8 ;; sizeof(f64)
    i32.mul
    local.get $base_row_sums
    i32.add

    ;; Store sum in rowSums[row]
    local.get $sum
    f64.store
  )

  ;; Import the exp function (host-provided)
)
