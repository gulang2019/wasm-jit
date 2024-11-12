cmake_minimum_required (VERSION 3.14)

project(wasm-vm)
set(CMAKE_BUILD_TYPE Debug)

include(vm_lib.cmake)

include_directories (inc . xbyak)

# --- Main executable --- #
add_executable (wasm-vm main.cpp)
target_link_libraries (wasm-vm vm -lm)

add_custom_command(
  TARGET wasm-vm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E create_symlink $<TARGET_FILE:wasm-vm> ../wasm-vm
)

