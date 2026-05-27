#include "gguf.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ESP_PLATFORM
#include <esp_partition.h>
#include <spi_flash_mmap.h>
#endif

gguf_str_t gguf_read_str(const uint8_t **cur)
{
    gguf_str_t s;
    s.len  = GGUF_READ(uint64_t, cur);
    s.data = (const char *)*cur;
    *cur += s.len;
    return s;
}

int gguf_str_eq(gguf_str_t s, const char *lit)
{
    size_t len = strlen(lit);
    return s.len == len && memcmp(s.data, lit, len) == 0;
}

uint64_t gguf_read_scalar(gguf_type type, const uint8_t **cur)
{
    switch (type)
    {
        case GGUF_UINT8: return GGUF_READ(uint8_t, cur);
        case GGUF_INT8: return (int8_t)GGUF_READ(uint8_t, cur);
        case GGUF_UINT16: return GGUF_READ(uint16_t, cur);
        case GGUF_INT16: return (int16_t)GGUF_READ(uint16_t, cur);
        case GGUF_UINT32: return GGUF_READ(uint32_t, cur);
        case GGUF_INT32: return (int32_t)GGUF_READ(uint32_t, cur);
        case GGUF_UINT64: return GGUF_READ(uint64_t, cur);
        case GGUF_INT64: return (int64_t)GGUF_READ(uint64_t, cur);
        case GGUF_FLOAT32: return GGUF_READ(uint32_t, cur);
        case GGUF_FLOAT64: return GGUF_READ(uint64_t, cur);
        case GGUF_BOOL: return GGUF_READ(uint8_t, cur);
        default: return 0;
    }
}

float gguf_read_float(gguf_type type, const uint8_t **cur)
{
    switch (type)
    {
        case GGUF_FLOAT32: return GGUF_READ(float, cur);
        case GGUF_FLOAT64: return (float)GGUF_READ(double, cur);
        default: return 0.0f;
    }
}

void gguf_skip_value(gguf_type type, const uint8_t **cur)
{
    if (type == GGUF_STRING)
    {
        gguf_read_str(cur);
        return;
    }
    if (type == GGUF_ARRAY)
    {
        uint32_t at = GGUF_READ(uint32_t, cur);
        uint64_t an = GGUF_READ(uint64_t, cur);
        for (uint64_t j = 0; j < an; j++) gguf_skip_value((gguf_type)at, cur);
        return;
    }
    gguf_read_scalar(type, cur);
}

void gguf_free(const void *buf)
{
#ifndef ESP_PLATFORM
    free((void *)buf);
#endif
}

char *gguf_load(const char *filename, const gguf_header_t **hdr, const uint8_t **data_start,
                const void **buf)
{
    const gguf_header_t *ptr;

#ifdef ESP_PLATFORM
    const esp_partition_t *partition =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "gguf");
    if (!partition) return "no partition";
    esp_partition_mmap_handle_t handle;
    esp_err_t err = esp_partition_mmap(partition, 0, partition->size, SPI_FLASH_MMAP_DATA,
                                       (void *)&ptr, &handle);
    if (err != ESP_OK) return "mmap failed";
#else
    FILE *f = fopen(filename, "rb");
    if (!f) return "fopen failed";
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    ptr = malloc(size);
    if (!ptr)
    {
        fclose(f);
        return "malloc failed";
    }
    if (fread((void *)ptr, 1, size, f) != size)
    {
        fclose(f);
        free((void *)ptr);
        return "fread failed";
    }
    fclose(f);
#endif

    if (ptr->magic != 0x46554747) return "bad magic";
    if (ptr->version != 3) return "unsupported version";

    *hdr        = ptr;
    *data_start = ptr->data;
    *buf        = (const void *)ptr;
    return NULL;
}

char *gguf_parse_meta(const uint8_t **cur, uint64_t n_kv, gguf_model_t *model, tokenizer_t *tok,
                      uint32_t *alignment)
{
    memset(model, 0, sizeof(*model));
    memset(tok, 0, sizeof(*tok));
    tok->unk_id = -1;
    tok->bos_id = -1;
    tok->eos_id = -1;

    for (uint64_t i = 0; i < n_kv; i++)
    {
        gguf_str_t key  = gguf_read_str(cur);
        gguf_type  type = GGUF_READ(uint32_t, cur);

        if (gguf_str_eq(key, "general.alignment"))
        {
            *alignment = (uint32_t)gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "general.architecture"))
        {
            model->architecture = gguf_read_str(cur);
        }
        else if (gguf_str_eq(key, "general.name"))
        {
            model->name = gguf_read_str(cur);
        }
        else if (gguf_str_eq(key, "llama.vocab_size"))
        {
            model->n_vocab = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.context_length"))
        {
            model->n_ctx = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.embedding_length"))
        {
            model->n_embd = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.feed_forward_length"))
        {
            model->n_ff = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.block_count"))
        {
            model->n_layer = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.attention.head_count"))
        {
            model->n_head = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.attention.head_count_kv"))
        {
            model->n_head_kv = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.attention.key_length"))
        {
            model->key_length = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.attention.value_length"))
        {
            model->value_length = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.rope.dimension_count"))
        {
            model->rope_dim = gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "llama.rope.freq_base"))
        {
            model->rope_freq_base = gguf_read_float(type, cur);
        }
        else if (gguf_str_eq(key, "llama.attention.layer_norm_rms_epsilon"))
        {
            model->rms_norm_eps = gguf_read_float(type, cur);
        }
        else if (gguf_str_eq(key, "tokenizer.ggml.tokens"))
        {
            GGUF_READ(uint32_t, cur);
            uint64_t n      = GGUF_READ(uint64_t, cur);
            tok->vocab_size = (int)n;
            tok->strings    = malloc(n * sizeof(char *));
            tok->lengths    = malloc(n * sizeof(int));
            if (!tok->strings || !tok->lengths)
            {
                free(tok->strings);
                free(tok->lengths);
                return "malloc";
            }
            for (uint64_t j = 0; j < n; j++)
            {
                tok->lengths[j] = GGUF_READ(uint64_t, cur);
                tok->strings[j] = (const char *)*cur;
                *cur += tok->lengths[j];
            }
        }
        else if (gguf_str_eq(key, "tokenizer.ggml.scores"))
        {
            GGUF_READ(uint32_t, cur);
            uint64_t n  = GGUF_READ(uint64_t, cur);
            tok->scores = malloc(n * sizeof(float));
            if (!tok->scores)
            {
                free(tok->strings);
                free(tok->lengths);
                return "malloc";
            }
            for (uint64_t j = 0; j < n; j++) tok->scores[j] = GGUF_READ(float, cur);
        }
        else if (gguf_str_eq(key, "tokenizer.ggml.token_type"))
        {
            GGUF_READ(uint32_t, cur);
            uint64_t n = GGUF_READ(uint64_t, cur);
            tok->types = malloc(n * sizeof(int));
            if (!tok->types)
            {
                free(tok->strings);
                free(tok->lengths);
                free(tok->scores);
                return "malloc";
            }
            for (uint64_t j = 0; j < n; j++) tok->types[j] = GGUF_READ(int32_t, cur);
        }
        else if (gguf_str_eq(key, "tokenizer.ggml.bos_token_id"))
        {
            tok->bos_id = (int)gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "tokenizer.ggml.eos_token_id"))
        {
            tok->eos_id = (int)gguf_read_scalar(type, cur);
        }
        else if (gguf_str_eq(key, "tokenizer.ggml.unknown_token_id"))
        {
            tok->unk_id = (int)gguf_read_scalar(type, cur);
        }
        else
        {
            gguf_skip_value(type, cur);
        }
    }

    if (model->n_vocab == 0 && tok->vocab_size > 0) model->n_vocab = tok->vocab_size;
    if (model->n_head_kv == 0) model->n_head_kv = model->n_head;

    return NULL;
}

char *gguf_read_tensors(const uint8_t **cur, uint64_t n, gguf_tensor_info_t *tensors,
                        uint32_t alignment, const void *buf)
{
    for (uint64_t i = 0; i < n; i++)
    {
        tensors[i].name   = gguf_read_str(cur);
        tensors[i].n_dims = GGUF_READ(uint32_t, cur);
        if (tensors[i].n_dims > 4) return "too many dims";
        for (uint32_t j = 0; j < tensors[i].n_dims; j++)
            tensors[i].dims[j] = GGUF_READ(uint64_t, cur);
        tensors[i].type   = GGUF_READ(uint32_t, cur);
        tensors[i].offset = GGUF_READ(uint64_t, cur);
    }

    uintptr_t off              = (uintptr_t)(*cur - (const uint8_t *)buf);
    off                        = (off + alignment - 1) & ~(uintptr_t)(alignment - 1);
    const uint8_t *tensor_data = (const uint8_t *)buf + off;
    for (uint64_t i = 0; i < n; i++) tensors[i].data = tensor_data + tensors[i].offset;

    return NULL;
}
