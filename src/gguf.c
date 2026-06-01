#include "gguf.h"
#include "platform.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if EMBEDLM_PLATFORM_ESP32S3
#include <esp_partition.h>
#include <spi_flash_mmap.h>
#endif

static char *gguf_finish_load(const gguf_header_t *ptr, const gguf_header_t **hdr,
                              const uint8_t **data_start, const void **buf)
{
    if (ptr->magic != 0x46554747) return "bad magic";
    if (ptr->version != 3) return "unsupported version";

    *hdr        = ptr;
    *data_start = ptr->data;
    *buf        = (const void *)ptr;
    return NULL;
}

#if EMBEDLM_PLATFORM_HOST || defined(EMBEDLM_SDCARD)
static char *gguf_load_file(const char *path, const gguf_header_t **hdr, const uint8_t **data_start,
                            const void **buf)
{
    FILE *f = fopen(path, "rb");
    if (!f) return "fopen failed";

    if (fseek(f, 0, SEEK_END) != 0)
    {
        fclose(f);
        return "fseek failed";
    }

    long end = ftell(f);
    if (end < 0)
    {
        fclose(f);
        return "ftell failed";
    }

    size_t size = (size_t)end;
    rewind(f);

    gguf_header_t *ptr = malloc(size);
    if (!ptr)
    {
        fclose(f);
        return "malloc failed";
    }
    if (fread((void *)ptr, 1, size, f) != size)
    {
        fclose(f);
        free(ptr);
        return "fread failed";
    }

    fclose(f);
    char *err = gguf_finish_load(ptr, hdr, data_start, buf);
    if (err) free(ptr);
    return err;
}
#endif

#if EMBEDLM_PLATFORM_STM32H7
static char *gguf_load_stm32(const char *filename, const gguf_header_t **hdr,
                             const uint8_t **data_start, const void **buf)
{
#if defined(EMBEDLM_SDCARD)
    char *err = gguf_load_file(filename, hdr, data_start, buf);
    if (!err) return NULL;

    if (!strchr(filename, '/'))
    {
        char        sd_path[128];
        const char *prefixes[] = {"0:/", "1:/", "/sdcard/"};
        for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++)
        {
            if (snprintf(sd_path, sizeof(sd_path), "%s%s", prefixes[i], filename) >=
                (int)sizeof(sd_path))
                continue;
            err = gguf_load_file(sd_path, hdr, data_start, buf);
            if (!err) return NULL;
        }
    }

    return err;
#else
    if (platform_embedded_model_start() == NULL || platform_embedded_model_end() == NULL ||
        platform_embedded_model_end() <= platform_embedded_model_start())
        return "embedded flash model missing";

    return gguf_finish_load((const gguf_header_t *)platform_embedded_model_start(), hdr, data_start,
                            buf);
#endif
}
#endif

static gguf_str_t gguf_read_str(const uint8_t **cur)
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

static uint64_t gguf_read_scalar(gguf_type type, const uint8_t **cur)
{
    switch (type)
    {
        case GGUF_UINT8: return GGUF_READ(uint8_t, cur);
        case GGUF_INT8: return (uint64_t)(int8_t)GGUF_READ(int8_t, cur);
        case GGUF_UINT16: return GGUF_READ(uint16_t, cur);
        case GGUF_INT16: return (uint64_t)(int16_t)GGUF_READ(int16_t, cur);
        case GGUF_UINT32: return GGUF_READ(uint32_t, cur);
        case GGUF_INT32: return (uint64_t)(int32_t)GGUF_READ(int32_t, cur);
        case GGUF_UINT64: return GGUF_READ(uint64_t, cur);
        case GGUF_INT64: return (uint64_t)GGUF_READ(int64_t, cur);
        case GGUF_FLOAT32: return GGUF_READ(uint32_t, cur);
        case GGUF_FLOAT64: return GGUF_READ(uint64_t, cur);
        case GGUF_BOOL: return GGUF_READ(uint8_t, cur);
        default: return 0;
    }
}

static float gguf_read_float(gguf_type type, const uint8_t **cur)
{
    switch (type)
    {
        case GGUF_FLOAT32: return GGUF_READ(float, cur);
        case GGUF_FLOAT64: return (float)GGUF_READ(double, cur);
        default: return 0.0f;
    }
}

static void gguf_skip_value(gguf_type type, const uint8_t **cur)
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
#if EMBEDLM_PLATFORM_STM32H7
    if (!platform_embedded_model_matches(buf)) free((void *)buf);
#elif EMBEDLM_PLATFORM_HOST
    free((void *)buf);
#endif
}

char *gguf_load(const char *filename, const gguf_header_t **hdr, const uint8_t **data_start,
                const void **buf)
{
    const gguf_header_t *ptr;

#if EMBEDLM_PLATFORM_ESP32S3
    const esp_partition_t *partition =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "gguf");
    if (!partition) return "no partition";
    esp_partition_mmap_handle_t handle;
    esp_err_t err = esp_partition_mmap(partition, 0, partition->size, SPI_FLASH_MMAP_DATA,
                                       (void *)&ptr, &handle);
    if (err != ESP_OK) return "mmap failed";
#elif EMBEDLM_PLATFORM_STM32H7
    return gguf_load_stm32(filename, hdr, data_start, buf);
#else
    return gguf_load_file(filename, hdr, data_start, buf);
#endif

    return gguf_finish_load(ptr, hdr, data_start, buf);
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
