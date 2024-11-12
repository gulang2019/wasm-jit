#include <stdexcept>

#include "CodeReader.h"
#include "common.h"
#include "ir.h"

Opcode_t CodePtr::rd_opcode() {
    return RD_OPCODE();
}

uint32_t CodePtr::rd_u32leb() {
    return read_u32leb(&buf);

}
int32_t CodePtr::rd_i32leb() {
    return read_i32leb(&buf);

}
uint64_t CodePtr::rd_u64leb() {
    return read_u64leb(&buf);
}
int64_t CodePtr::rd_i64leb() {
    return read_i64leb(&buf);

}
uint8_t CodePtr::rd_u8() {
    return read_u8(&buf);

}
uint32_t CodePtr::rd_u32() {
    return read_u32(&buf);

}
uint64_t CodePtr::rd_u64() {
    return read_u64(&buf);

}

void CodePtr::skip_leb() {
    while (buf.ptr != buf.end) {
		auto b = *buf.ptr++;
        if ((b & 0x80) == 0) break;
	}
}

void CodePtr::skip_block_type(){
    auto code = rd_u32leb();
    // TODO: if (BpConstants.typeCodeHasIndex(code)) skip_leb();
}

std::vector<int> CodePtr::rd_labels() {
    int length =rd_u32leb();
    std::vector<int> labels;
    for (int i = 0; i <= length; ++i)
        labels.push_back(rd_u32leb());
    return labels;  
}

MemArg CodePtr::rd_mem_arg(){
    return {
        .flags = rd_u32leb(),
        .memory_index = 0,
        .offset = rd_u64leb()
    };
}