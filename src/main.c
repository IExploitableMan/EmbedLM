#include <stdint.h>
#include <stdio.h>
#ifdef ESP_PLATFORM
#include <esp_partition.h>
#include <spi_flash_mmap.h>
#else
#include <stdlib.h>
#endif

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
} gguf_type_t;

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
        &ptr,
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

void print_gguf_value(gguf_type_t type, const uint8_t **cur)
{
    switch (type) {
        case GGUF_UINT8:   printf("%u", READ(uint8_t,  cur)); break;
        case GGUF_INT8:    printf("%d", READ(int8_t,   cur)); break;
        case GGUF_UINT16:  printf("%u", READ(uint16_t, cur)); break;
        case GGUF_INT16:   printf("%d", READ(int16_t,  cur)); break;
        case GGUF_UINT32:  printf("%u", READ(uint32_t, cur)); break;
        case GGUF_INT32:   printf("%d", READ(int32_t,  cur)); break;
        case GGUF_UINT64:  printf("%llu", (unsigned long long)READ(uint64_t, cur)); break;
        case GGUF_INT64:   printf("%lld", (long long)READ(int64_t, cur)); break;
        case GGUF_FLOAT32: printf("%f", READ(float,    cur)); break;
        case GGUF_FLOAT64: printf("%lf", READ(double,  cur)); break;
        case GGUF_BOOL:    printf("%s", READ(uint8_t, cur) ? "true" : "false"); break;

        case GGUF_STRING: {
            gguf_string_t s = read_gguf_string(cur);
            printf("%.*s", (int)s.len, s.data);
            break;
        }

        case GGUF_ARRAY: {
            gguf_type_t elem_type = READ(uint32_t, cur);
            uint64_t count = READ(uint64_t, cur);

            printf("[");
            for (uint64_t i = 0; i < count; i++) {
                if (i) printf(", ");
                print_gguf_value(elem_type, cur);
            }
            printf("]");
            break;
        }

        default:
            printf("<unknown>");
            break;
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
    for (uint64_t i = 0; i < gguf->metadata_kv_count; i++) {
        gguf_string_t key = read_gguf_string(&cur);
        gguf_type_t type = READ(uint32_t, &cur);

        if (strcmp_key(key, "general.alignment")) {
            alignment = READ(uint32_t, &cur);
            continue;
        }

        printf("%.*s = ", (int)key.len, key.data);
        print_gguf_value(type, &cur);
        printf("\n");
    }
    for (uint64_t i = 0; i < gguf->tensor_count; i++) {
        gguf_string_t name = read_gguf_string(&cur);
        uint32_t n_dims = READ(uint32_t, &cur);

        uint64_t dims[4] = {0};
        for (uint32_t j = 0; j < n_dims; j++)
            dims[j] = READ(uint64_t, &cur);

        uint32_t type = READ(uint32_t, &cur);
        uint64_t offset = READ(uint64_t, &cur);

        printf("%.*s: ", (int)name.len, name.data);
        printf("dims=[");
        for (uint32_t j = 0; j < n_dims; j++) {
            if (j) printf(", ");
            printf("%llu", (unsigned long long)dims[j]);
        }
        printf("] type=%u offset=%llu\n",
            type,
            (unsigned long long)offset);
        const uint8_t *tensor_data = (const uint8_t*)(((uintptr_t)cur + alignment - 1) & ~(uintptr_t)(alignment - 1));
    }
}

#ifndef ESP_PLATFORM
int main(void) { app_main(); }
#endif
