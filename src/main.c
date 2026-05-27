#include "gguf.h"
#include "infer.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef ESP_PLATFORM
#include <esp_task_wdt.h>
#include <esp_timer.h>
#endif

static uint64_t now_us(void)
{
#ifdef ESP_PLATFORM
    return esp_timer_get_time();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
#endif
}

#define MAX_TOKENS 200

void app_main(void)
{
#ifdef ESP_PLATFORM
    esp_task_wdt_config_t twdt = {.timeout_ms = 120000, .trigger_panic = false};
    esp_task_wdt_reconfigure(&twdt);
    esp_task_wdt_add(NULL);
#endif
    const gguf_header_t *gguf;
    const uint8_t       *cur;
    const void          *gguf_buf;

    const char *err = gguf_load("large.gguf", &gguf, &cur, &gguf_buf);
    if (err)
    {
        printf("error: gguf load: %s\n", err);
        return;
    }

    gguf_model_t model;
    tokenizer_t  tok;
    uint32_t     alignment = 32;

    err = gguf_parse_meta(&cur, gguf->metadata_kv_count, &model, &tok, &alignment);
    if (err)
    {
        printf("error: parse: %s\n", err);
        gguf_free(gguf_buf);
        return;
    }

    if (!gguf_str_eq(model.architecture, "llama"))
    {
        printf("error: unsupported arch\n");
        gguf_free(gguf_buf);
        return;
    }

    gguf_tensor_info_t *tensors = malloc(gguf->tensor_count * sizeof(*tensors));
    if (!tensors)
    {
        printf("error: tensor alloc\n");
        return;
    }

    err = gguf_read_tensors(&cur, gguf->tensor_count, tensors, alignment, gguf_buf);
    if (err)
    {
        printf("error: tensor read: %s\n", err);
        free(tensors);
        return;
    }

    llama_model_t m;
    if (llama_build_model(&m, &model, tensors, gguf->tensor_count) != 0)
    {
        printf("error: build model\n");
        free(tensors);
        return;
    }

    if (tokenizer_init(&tok) != 0)
    {
        printf("error: tokenizer init\n");
        llama_free_model(&m);
        free(tensors);
        return;
    }

    llama_scratch_t scratch;
    if (llama_scratch_init(&scratch, &m) != 0)
    {
        printf("error: scratch alloc\n");
        tokenizer_free(&tok);
        llama_free_model(&m);
        free(tensors);
        return;
    }

    kv_cache_t cache;
    if (kv_cache_init(&cache, m.n_layer, m.n_head_kv, m.head_dim, LLAMA_MAX_CTX) != 0)
    {
        printf("error: kv cache init\n");
        llama_scratch_free(&scratch);
        tokenizer_free(&tok);
        llama_free_model(&m);
        free(tensors);
        return;
    }

    printf("n_layer=%d n_embd=%d n_head=%d n_head_kv=%d n_ff=%d n_vocab=%d head_dim=%d\n",
           m.n_layer, m.n_embd, m.n_head, m.n_head_kv, m.n_ff, m.n_vocab, m.head_dim);

    float *logits = malloc(m.n_vocab * sizeof(float));
    if (!logits)
    {
        printf("error: logits alloc\n");
        kv_cache_free(&cache);
        llama_scratch_free(&scratch);
        tokenizer_free(&tok);
        llama_free_model(&m);
        free(tensors);
        return;
    }

    int         prompt_ids[256];
    const char *prompt_text = "Once upon a time";
    int         n_prompt    = tokenizer_encode(&tok, prompt_text, prompt_ids, 256);
    if (n_prompt <= 0)
    {
        printf("error: tokenize prompt\n");
        free(logits);
        kv_cache_free(&cache);
        llama_scratch_free(&scratch);
        tokenizer_free(&tok);
        llama_free_model(&m);
        free(tensors);
        return;
    }

    printf("%s", prompt_text);
    fflush(stdout);

    for (int pos = 0; pos < n_prompt; pos++)
    {
        llama_forward(&m, &cache, &scratch, prompt_ids[pos], pos, logits);
#ifdef ESP_PLATFORM
        esp_task_wdt_reset();
#endif
    }

    char text[4096];
    srand((unsigned)time(NULL));
    int      token;
    uint64_t t_start = now_us();
    uint64_t ttft    = 0;
    int      n_gen   = 0;
    for (int pos = n_prompt; pos < LLAMA_MAX_CTX && n_gen < MAX_TOKENS; pos++)
    {
        token = sample(logits, m.n_vocab, 0.8f, 40);
        if (token == tok.eos_id) break;
        tokenizer_decode(&tok, &token, 1, text, sizeof(text));
        printf("%s", text);
        fflush(stdout);
        if (n_gen == 0) ttft = now_us() - t_start;
        n_gen++;
        llama_forward(&m, &cache, &scratch, token, pos, logits);
#ifdef ESP_PLATFORM
        esp_task_wdt_reset();
#endif
    }
    uint64_t total_us = now_us() - t_start;
    float    tps      = (float)n_gen / ((float)total_us / 1e6f);
    printf("\n(%.2f tok/sec) (%d tokens) (%llums)\n", (double)tps, n_gen,
           (unsigned long long)ttft / 1000);

    free(logits);
    kv_cache_free(&cache);
    llama_scratch_free(&scratch);
    tokenizer_free(&tok);
    llama_free_model(&m);
    free(tensors);
    gguf_free(gguf_buf);
}

#ifndef ESP_PLATFORM
int main(void)
{
    app_main();
    return 0;
}
#endif /* ESP_PLATFORM */
