(module
  (func (export "main") (param i32) (result i32)
    (local i32)
    i32.const -22
    memory.size
    local.set 1
    (memory.grow (local.get 0))
    local.get 1
    i32.ne
    br_if 0
    drop
    (i32.store (i32.const 131584) (i32.const 63))
    (i32.load (i32.const 131584))
    memory.size
    i32.add
  )
  (memory (;0;) 2 5)
  (data (;0;) (i32.const 65536) "\13\00\00\00\05\00\00\00\1c\00\01\00$\00\01\00,\00\01\004\00\01\00<\00\01\00\00\00\00\00\0b\00\00\00\00\00\00\00\0c\00\00\00\00\00\00\00!\00\00\00\00\00\00\00,\00\00\00\00\00\00\00\fd&\ff\ff")
)
