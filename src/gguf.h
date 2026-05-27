#ifndef GGUF_H
#define GGUF_H

#include <stddef.h>
#include <stdint.h>
#include "tokenizer.h"

typedef enum
{
    GGML_F32     = 0,
    GGML_F16     = 1,
    GGML_Q4_0    = 2,
    GGML_Q4_1    = 3,
    GGML_Q5_0    = 6,
    GGML_Q5_1    = 7,
    GGML_Q8_0    = 8,
    GGML_Q8_1    = 9,
    GGML_Q2_K    = 10,
    GGML_Q3_K    = 11,
    GGML_Q4_K    = 12,
    GGML_Q5_K    = 13,
    GGML_Q6_K    = 14,
    GGML_Q8_K    = 15,
    GGML_IQ2_XXS = 16,
    GGML_IQ2_XS  = 17,
    GGML_IQ3_XXS = 18,
    GGML_IQ1_S   = 19,
    GGML_IQ4_NL  = 20,
    GGML_IQ3_S   = 21,
    GGML_IQ2_S   = 22,
    GGML_IQ4_XS  = 23,
    GGML_I8      = 24,
    GGML_I16     = 25,
    GGML_I32     = 26,
    GGML_I64     = 27,
    GGML_F64     = 28,
    GGML_IQ1_M   = 29,
    GGML_BF16    = 30,
    GGML_TQ1_0   = 34,
    GGML_TQ2_0   = 35,
    GGML_MXFP4   = 39,
} ggml_type;

typedef enum
{
    GGUF_UINT8   = 0,
    GGUF_INT8    = 1,
    GGUF_UINT16  = 2,
    GGUF_INT16   = 3,
    GGUF_UINT32  = 4,
    GGUF_INT32   = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL    = 7,
    GGUF_STRING  = 8,
    GGUF_ARRAY   = 9,
    GGUF_UINT64  = 10,
    GGUF_INT64   = 11,
    GGUF_FLOAT64 = 12,
} gguf_type;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    uint8_t  data[];
} gguf_header_t;

typedef struct {
    uint64_t    len;
    const char *data;
} gguf_str_t;

typedef struct {
    gguf_str_t architecture;
    gguf_str_t name;
    uint64_t   n_vocab;
    uint64_t   n_ctx;
    uint64_t   n_embd;
    uint64_t   n_ff;
    uint64_t   n_layer;
    uint64_t   n_head;
    uint64_t   n_head_kv;
    uint64_t   key_length;
    uint64_t   value_length;
    uint64_t   rope_dim;
    float      rope_freq_base;
    float      rms_norm_eps;
} gguf_model_t;

typedef struct {
    gguf_str_t  name;
    uint32_t    n_dims;
    uint64_t    dims[4];
    ggml_type   type;
    uint64_t    offset;
    const void *data;
} gguf_tensor_info_t;

#define GGUF_READ(type, p)                                                                         \
    __extension__({                                                                                \
        type v;                                                                                    \
        memcpy(&v, *(p), sizeof(type));                                                            \
        *(p) += sizeof(type);                                                                      \
        v;                                                                                         \
    })

gguf_str_t gguf_read_str(const uint8_t **cur);
int        gguf_str_eq(gguf_str_t s, const char *lit);
uint64_t   gguf_read_scalar(gguf_type type, const uint8_t **cur);
float      gguf_read_float(gguf_type type, const uint8_t **cur);
void       gguf_skip_value(gguf_type type, const uint8_t **cur);

void  gguf_free(const void *buf);
char *gguf_load(const char *filename, const gguf_header_t **hdr, const uint8_t **data_start,
                const void **buf);
char *gguf_parse_meta(const uint8_t **cur, uint64_t n_kv, gguf_model_t *model, tokenizer_t *tok,
                      uint32_t *alignment);
char *gguf_read_tensors(const uint8_t **cur, uint64_t n, gguf_tensor_info_t *tensors,
                        uint32_t alignment, const void *buf);

#endif /* GGUF_H */
