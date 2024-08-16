#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <string>
#include <array>
#include <list>
#include <vector>
#include <deque>

/*** Generic used typedefs ***/
typedef uint8_t byte;
typedef std::vector<byte> bytearr;
typedef std::deque<byte> bytedeque;
typedef uint32_t Opcode_t;
/* Buffer for parsing/decoding */
typedef struct {
  const byte* start;
  const byte* ptr;
  const byte* end;
} buffer_t;
/***************/

extern int g_threads;
/*** Global trace/err/disassemble flags and macros. ***/
#define TRACE(...) do { if(g_trace) fprintf(stderr, __VA_ARGS__); } while(0)
#define ERR(...) fprintf(stderr, __VA_ARGS__)
extern int g_trace;
/***************/

extern int g_time;

/*** Parsing macros ***/
#define RD_U32()        read_u32leb(&buf)
#define RD_I32()        read_i32leb(&buf)
#define RD_U64()        read_u64leb(&buf)
#define RD_I64()        read_i64leb(&buf)
#define RD_NAME()       read_name(&buf)
#define RD_BYTESTR(len) read_bytes(&buf, len)

#define RD_BYTE()       read_u8(&buf)
#define RD_U32_RAW()    read_u32(&buf)
#define RD_U64_RAW()    read_u64(&buf)


#define VALIDATE_OP(b) {  \
  opcode_entry_t op_entry = opcode_table[b]; \
  if (op_entry.invalid || (op_entry.mnemonic == 0)) { \
    ERR("Unimplemented opcode %d: %s\n", b, op_entry.mnemonic);  \
    throw std::runtime_error("Unimplemented");  \
  } \
}

#define RD_U32_RAW_BE() ({ \
  uint32_t res = 0; \
  byte opclass; \
  byte by = opclass = res = RD_BYTE();  \
  VALIDATE_OP(opclass); \
  if ((opclass >= 0xFB) && (opclass <= 0xFE)) { \
    while (by & 0x80) {  \
      by = RD_BYTE(); \
      res = ((res << 8) + by);  \
    } \
  } \
  res;  \
})


#define RD_OPCODE() ({  \
  Opcode_t lb = RD_U32_RAW_BE();  \
  VALIDATE_OP(lb);  \
  lb; \
})
/********************/


/*** List macros ***/
#define GET_LIST_ELEM(ll, idx) ({ \
  auto it = std::next(ll.begin(), idx); \
  &(*it);  \
})

#define GET_LIST_IDX(ll, ptr) ({  \
  auto it = ll.begin(); \
  uint32_t i = 0; \
  for (it = ll.begin(); it != ll.end(); ++it, ++i) {  \
    if (&(*it) == ptr) { break; } \
  } \
  i;  \
})

#define GET_DEQUE_ELEM(deq, idx) ({ &deq[idx]; })
#define GET_DEQUE_IDX(deq, ptr) ({ \
    auto res = std::find_if(deq.begin(), deq.end(), [ptr](auto &i) { return ptr == &i; });  \
    std::distance(deq.begin(), res); \
    })
/********************/


/* Load a file into memory; initialize start and end.
* Returns < 0 on failure. */
ssize_t load_file(const char* path, byte** start, byte** end);

/* Unload a file previously loaded into memory using {load_file}. */
ssize_t unload_file(byte** start, byte** end);


/*** Decode Operations ***/

#define MAX_FILE_SIZE 2000000000

/* Decode LEB128 values, enforcing the specified signedness and maximum range.
* The argument {ptr} will be advanced to point after the decoded value.
* If an error occurs (e.g. unterminated LEB, invalid value), {ptr} will be set to
* NULL. */
int32_t  decode_i32leb(const byte* ptr, const byte* limit, ssize_t* len);
uint32_t decode_u32leb(const byte* ptr, const byte* limit, ssize_t* len);
int64_t  decode_i64leb(const byte* ptr, const byte* limit, ssize_t* len);
uint64_t decode_u64leb(const byte* ptr, const byte* limit, ssize_t* len);

/* Decode fixed-size integer values. */
uint32_t decode_u32(const byte* ptr, const byte* limit, ssize_t* len);



/* Read an unsigned(u)/signed(i) X-bit LEB, advancing the {ptr} in buffer */
uint32_t read_u32leb(buffer_t* buf);
int32_t read_i32leb(buffer_t* buf);
uint64_t read_u64leb(buffer_t* buf);
int64_t read_i64leb(buffer_t* buf);

/* Read a raw X-bit value, advancing the {ptr} in buffer*/
uint8_t read_u8(buffer_t* buf);
uint32_t read_u32(buffer_t* buf);
uint64_t read_u64(buffer_t* buf);

/* Read a name (32leb + string), advancing the {ptr} in buffer */
std::string read_name(buffer_t* buf);
/* Read num_bytes, advancing the {ptr} in buffer */
bytearr read_bytes(buffer_t* buf, uint32_t num_bytes);

