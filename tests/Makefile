# Directories
CTESTS_DIR = ctests
WATTSTS_DIR = wattests
OUTPUT_DIR = wasm
OUTPUT_EC_DIR = wasm-ec

# Tools
CLANG = /opt/wasi-sdk/bin/clang
WAT2WASM = wat2wasm
WASM2WAT = wasm2wat

# File extensions
C_EXT = .c
WAT_EXT = .wat
WASM_EXT = .wasm

# Find all C files and WAT files
C_FILES = $(wildcard $(CTESTS_DIR)/*$(C_EXT))
WAT_FILES = $(wildcard $(WATTSTS_DIR)/*$(WAT_EXT))

# Define WASM files from C and WAT sources
WASM_FILES_FROM_C = $(patsubst $(CTESTS_DIR)/%$(C_EXT),$(OUTPUT_DIR)/%$(WASM_EXT),$(C_FILES))
WASM_FILES_FROM_WAT = $(patsubst $(WATTSTS_DIR)/%$(WAT_EXT),$(OUTPUT_DIR)/%$(WASM_EXT),$(WAT_FILES))

WASM_FILES = $(filter-out $(OUTPUT_DIR)/ec_%, $(WASM_FILES_FROM_C)) $(filter-out $(OUTPUT_DIR)/ec_%, $(WASM_FILES_FROM_WAT))
WASM_EC_FILES = $(patsubst $(OUTPUT_DIR)/%, $(OUTPUT_EC_DIR)/%, $(filter $(OUTPUT_DIR)/ec_%, $(WASM_FILES_FROM_C)) $(filter $(OUTPUT_DIR)/ec_%, $(WASM_FILES_FROM_WAT)))

# All targets
all: $(WASM_FILES) $(WASM_EC_FILES)

# Rule to build WASM from C files
$(OUTPUT_DIR)/%$(WASM_EXT): $(CTESTS_DIR)/%$(C_EXT)
	@mkdir -p $(OUTPUT_DIR)
	$(CLANG) --target=wasm32 -O1 -nostartfiles -nostdlib -Wl,--no-entry -o $@ $<

# Rule to convert WAT files to WASM
$(OUTPUT_DIR)/%$(WASM_EXT): $(WATTSTS_DIR)/%$(WAT_EXT)
	@mkdir -p $(OUTPUT_DIR)
	$(WAT2WASM) $< -o $@

# Rule to build WASM from C files
$(OUTPUT_EC_DIR)/%$(WASM_EXT): $(CTESTS_DIR)/%$(C_EXT)
	@mkdir -p $(OUTPUT_EC_DIR)
	$(CLANG) --target=wasm32 -O0 -nostartfiles -nostdlib -Wl,--no-entry -o $@ $<

# Rule to convert WAT files to WASM
$(OUTPUT_EC_DIR)/%$(WASM_EXT): $(WATTSTS_DIR)/%$(WAT_EXT)
	@mkdir -p $(OUTPUT_EC_DIR)
	$(WAT2WASM) $< -o $@

# Clean up generated files
clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(OUTPUT_EC_DIR)

.PHONY: all clean

