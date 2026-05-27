#include "gguf.h"
#include "infer.h"
#include "platform.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TOKENS 200

void app_main(void)
{
    platform_init();
    platform_watchdog_init();

    const char *model_path = platform_default_model_path();
    const char *err        = NULL;

#if EMBEDLM_PLATFORM_HOST
    const char *host_model_path = getenv("EMBEDLM_MODEL_PATH");
    if (host_model_path && host_model_path[0] != '\0') model_path = host_model_path;
#endif

    const gguf_header_t *gguf     = NULL;
    const uint8_t       *cur      = NULL;
    const void          *gguf_buf = NULL;
    gguf_tensor_info_t  *tensors  = NULL;
    float               *logits   = NULL;

    gguf_model_t    model;
    tokenizer_t     tok;
    llama_model_t   m;
    llama_scratch_t scratch;
    kv_cache_t      cache;

    memset(&model, 0, sizeof(model));
    memset(&tok, 0, sizeof(tok));
    memset(&m, 0, sizeof(m));
    memset(&scratch, 0, sizeof(scratch));
    memset(&cache, 0, sizeof(cache));

    err = gguf_load(model_path, &gguf, &cur, &gguf_buf);
    if (err)
    {
        printf("error: gguf load: %s\n", err);
        goto cleanup;
    }

    uint32_t alignment = 32;

    err = gguf_parse_meta(&cur, gguf->metadata_kv_count, &model, &tok, &alignment);
    if (err)
    {
        printf("error: parse: %s\n", err);
        goto cleanup;
    }

    if (!gguf_str_eq(model.architecture, "llama"))
    {
        printf("error: unsupported arch\n");
        goto cleanup;
    }

    tensors = malloc(gguf->tensor_count * sizeof(*tensors));
    if (!tensors)
    {
        printf("error: tensor alloc\n");
        goto cleanup;
    }

    err = gguf_read_tensors(&cur, gguf->tensor_count, tensors, alignment, gguf_buf);
    if (err)
    {
        printf("error: tensor read: %s\n", err);
        goto cleanup;
    }

    if (llama_build_model(&m, &model, tensors, gguf->tensor_count) != 0)
    {
        printf("error: build model\n");
        goto cleanup;
    }

    if (tokenizer_init(&tok) != 0)
    {
        printf("error: tokenizer init\n");
        goto cleanup;
    }

    if (llama_scratch_init(&scratch, &m) != 0)
    {
        printf("error: scratch alloc\n");
        goto cleanup;
    }

    if (kv_cache_init(&cache, m.n_layer, m.n_head_kv, m.head_dim, LLAMA_MAX_CTX) != 0)
    {
        printf("error: kv cache init\n");
        goto cleanup;
    }

    printf("n_layer=%d n_embd=%d n_head=%d n_head_kv=%d n_ff=%d n_vocab=%d head_dim=%d\n",
           m.n_layer, m.n_embd, m.n_head, m.n_head_kv, m.n_ff, m.n_vocab, m.head_dim);

    logits = malloc(m.n_vocab * sizeof(float));
    if (!logits)
    {
        printf("error: logits alloc\n");
        goto cleanup;
    }

    int         prompt_ids[256];
    const char *prompt_text = "Once upon a time";
    int         n_prompt    = tokenizer_encode(&tok, prompt_text, prompt_ids, 256);
    if (n_prompt <= 0)
    {
        printf("error: tokenize prompt\n");
        goto cleanup;
    }

    printf("%s", prompt_text);
    fflush(stdout);

    for (int pos = 0; pos < n_prompt; pos++)
    {
        llama_forward(&m, &cache, &scratch, prompt_ids[pos], pos, logits);
        platform_watchdog_kick();
    }

    char text[4096];
    srand((unsigned)time(NULL));
    int      token;
    uint64_t t_start = platform_now_us();
    uint64_t ttft    = 0;
    int      n_gen   = 0;
    for (int pos = n_prompt; pos < LLAMA_MAX_CTX && n_gen < MAX_TOKENS; pos++)
    {
        token = sample(logits, m.n_vocab, 0.8f, 40);
        if (token == tok.eos_id) break;
        tokenizer_decode(&tok, &token, 1, text, sizeof(text));
        printf("%s", text);
        fflush(stdout);
        if (n_gen == 0) ttft = platform_now_us() - t_start;
        n_gen++;
        llama_forward(&m, &cache, &scratch, token, pos, logits);
        platform_watchdog_kick();
    }
    uint64_t total_us = platform_now_us() - t_start;
    float    tps      = (float)n_gen / ((float)total_us / 1e6f);
    printf("\n(%.2f tok/sec) (%d tokens) (%llums)\n", (double)tps, n_gen,
           (unsigned long long)ttft / 1000);

cleanup:
    free(logits);
    kv_cache_free(&cache);
    llama_scratch_free(&scratch);
    tokenizer_free(&tok);
    llama_free_model(&m);
    free(tensors);
    if (gguf_buf) gguf_free(gguf_buf);
}

#if !EMBEDLM_PLATFORM_ESP32S3
int main(void)
{
    app_main();
    return 0;
}
#endif /* !EMBEDLM_PLATFORM_ESP32S3 */
