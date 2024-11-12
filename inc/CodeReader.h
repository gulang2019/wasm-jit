#pragma once 

#include "common.h"
#include "instance.h"

#include <vector>

class CodePtr {
    buffer_t buf;

public:

    void reset(const bytearr* code, int npos, int nlimit) {
		if (code){
			buf.start = code->data();
			buf.ptr = code->data() + npos;
			buf.end = code->data() + nlimit;
		}
		else {
			buf.start = buf.ptr = buf.end = nullptr;
		}
	}

	Opcode_t rd_opcode();
	/* Read an unsigned(u)/signed(i) X-bit LEB, advancing the {ptr} in buffer */
	uint32_t rd_u32leb();
	int32_t rd_i32leb();
	uint64_t rd_u64leb();
	int64_t rd_i64leb();

	/* Read a raw X-bit value, advancing the {ptr} in buffer*/
	uint8_t rd_u8();
	uint32_t rd_u32();
	uint64_t rd_u64();

	inline bool is_end() {return buf.ptr == buf.end;}
	inline int offset() {return buf.ptr - buf.start;}
	inline void at(int npos) {buf.ptr = buf.start + npos;}

	void skip_block_type();
	void skip_leb();
	MemArg rd_mem_arg();

	std::vector<int> rd_labels();
};