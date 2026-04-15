#include <stdint.h>
#include <stdio.h>
#ifdef ESP_PLATFORM
#include <esp_partition.h>
#include <spi_flash_mmap.h>
#else
#include <stdlib.h>
#endif
#include <string.h>

typedef enum {
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

typedef enum {
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

typedef struct
{
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    uint8_t data[];
} gguf_t;

typedef struct {
    uint64_t len;
    const char *data;
} gguf_string_t;

typedef struct {
    gguf_string_t name;
    uint32_t n_dims;
    uint64_t dims[4];
    ggml_type type;
    uint64_t offset;
} gguf_tensor_t;

typedef struct {
    gguf_string_t architecture;
    gguf_string_t name;
    uint64_t n_vocab;
    uint64_t n_ctx;
    uint64_t n_embd;
    uint64_t n_ff;
    uint64_t n_layer;
    uint64_t n_head;
    uint64_t n_head_kv;
    uint64_t key_length;
    uint64_t value_length;
    uint64_t rope_dim;
    float rope_freq_base;
    float rms_norm_eps;
} gguf_model_t;

// TODO: fix unaligned access
#define READ(type, p) ({ \
    type v = *(const type*)(*(p)); \
    *(p) += sizeof(type); \
    v; \
})

char *load_gguf(const gguf_t **gguf)
{
    const gguf_t *ptr;

#ifdef ESP_PLATFORM
    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        0x40,
        "gguf");
    if (!partition)
        return "no partition";

    esp_partition_mmap_handle_t handle;
    esp_err_t err = esp_partition_mmap(
        partition,
        0,
        partition->size,
        SPI_FLASH_MMAP_DATA,
        (void*)&ptr,
        &handle);
    if (err != ESP_OK)
        return "mmap failed";
#else
    FILE *f = fopen("gguf.gguf", "rb");
    if (!f) return "fopen failed";

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    ptr = malloc(size);
    if (!ptr) {
        fclose(f);
        return "malloc failed";
    }

    if (fread((void*)ptr, 1, size, f) != size) {
        fclose(f);
        free((void*)ptr);
        return "fread failed";
    }

    fclose(f);
#endif

    if (ptr->magic != 0x46554747)
        return "bad magic";
    if (ptr->version != 3)
        return "unsupported version";

    *gguf = ptr;
    return NULL;
}

gguf_string_t read_gguf_string(const uint8_t **cur)
{
    gguf_string_t str;
    str.len = READ(uint64_t, cur);
    str.data = (const char*)*cur;
    *cur += str.len;
    return str;
}

static int compare_gguf_string(gguf_string_t str, const char *lit)
{
    size_t len = strlen(lit);
    return str.len == len && memcmp(str.data, lit, len) == 0;
}

void skip_gguf_value(gguf_type type, const uint8_t **cur)
{
    switch (type) {
        case GGUF_UINT8:   READ(uint8_t,  cur);   break;
        case GGUF_INT8:    READ(int8_t,   cur);   break;
        case GGUF_UINT16:  READ(uint16_t, cur);   break;
        case GGUF_INT16:   READ(int16_t,  cur);   break;
        case GGUF_UINT32:  READ(uint32_t, cur);   break;
        case GGUF_INT32:   READ(int32_t,  cur);   break;
        case GGUF_UINT64:  READ(uint64_t, cur);   break;
        case GGUF_INT64:   READ(int64_t,  cur);   break;
        case GGUF_FLOAT32: READ(float,    cur);   break;
        case GGUF_FLOAT64: READ(double,   cur);   break;
        case GGUF_BOOL:    READ(uint8_t,  cur);   break;
        case GGUF_STRING:  read_gguf_string(cur); break;
        case GGUF_ARRAY: {
            for (uint64_t i = 0; i < READ(uint64_t, cur); i++) 
                skip_gguf_value(READ(uint32_t, cur), cur);
            break;
        }
    }
}

void app_main(void)
{
    const gguf_t *gguf;
    const char *err = load_gguf(&gguf);
    if (err)
    {
        printf("GGUF load failed: %s\n", err);
        return;
    }
    const uint8_t *cur = gguf->data;
    uint32_t alignment = 32;
    gguf_model_t *model = calloc(1, sizeof(*model)); 
    for (uint64_t i = 0; i < gguf->metadata_kv_count; i++) {
        gguf_string_t key = read_gguf_string(&cur);
        gguf_type type = READ(uint32_t, &cur);

        if (compare_gguf_string(key, "general.alignment")) {
            alignment = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "general.architecture")) {
            model->architecture = read_gguf_string(&cur);
        }
        else if (compare_gguf_string(key, "general.name")) {
            model->name = read_gguf_string(&cur);
        }
        else if (compare_gguf_string(key, "llama.vocab_size")) {
            model->n_vocab = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.context_length")) {
            model->n_ctx = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.embedding_length")) {
            model->n_embd = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.feed_forward_length")) {
            model->n_ff = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.block_count")) {
            model->n_layer = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.attention.head_count")) {
            model->n_head = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.attention.head_count_kv")) {
            model->n_head_kv = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.attention.key_length")) {
            model->key_length = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.attention.value_length")) {
            model->value_length = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.rope.dimension_count")) {
            model->rope_dim = READ(uint32_t, &cur);
        }
        else if (compare_gguf_string(key, "llama.rope.freq_base")) {
            model->rope_freq_base = READ(float, &cur);
        }
        else if (compare_gguf_string(key, "llama.attention.layer_norm_rms_epsilon")) {
            model->rms_norm_eps = READ(float, &cur);
        }
        else {
            skip_gguf_value(type, &cur);
        }
    }
    if (!compare_gguf_string(model->architecture, "llama")) {
        printf("unsupported arch\n");
        return;
    }
    gguf_tensor_t *tensors = malloc(sizeof(*tensors) * gguf->tensor_count);
    if (!tensors) {
        printf("tensor alloc failed\n");
        return;
    }
    for (uint64_t i = 0; i < gguf->tensor_count; i++) {
        gguf_tensor_t *tensor = &tensors[i];

        tensor->name = read_gguf_string(&cur);

        tensor->n_dims = READ(uint32_t, &cur);
        if (tensor->n_dims > 4) {
            printf("too many dims\n");
            return;
        }
        for (uint32_t j = 0; j < tensor->n_dims; j++)
            tensor->dims[j] = READ(uint64_t, &cur);

        tensor->type = READ(uint32_t, &cur);
        tensor->offset = READ(uint64_t, &cur);
    }
    const uint8_t *tensor_data = (const uint8_t*)(((uintptr_t)cur + alignment - 1) & ~(uintptr_t)(alignment - 1));
}

#ifndef ESP_PLATFORM
int main(void) { app_main(); }
#endif
