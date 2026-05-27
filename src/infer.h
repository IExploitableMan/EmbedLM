#ifndef INFER_H
#define INFER_H

#include "gguf.h"
#include <stdint.h>

#define LLAMA_MAX_CTX 1024

typedef struct {
    float *x;
    float *buf;
    float *attn_out;
    float *q;
    float *k;
    float *v;
    float *gate;
    float *up;
    float *score;
} llama_scratch_t;

typedef struct {
    int   n_layer;
    int   n_embd;
    int   n_head;
    int   n_head_kv;
    int   n_ff;
    int   n_vocab;
    int   head_dim;
    float rope_freq_base;
    float rms_norm_eps;

    const float   *output_norm;
    const uint8_t *token_embd;
    const uint8_t *output_weight;

    struct {
        const float   *attn_norm;
        const uint8_t *wq;
        const uint8_t *wk;
        const uint8_t *wv;
        const uint8_t *wo;
        const float   *ffn_norm;
        const uint8_t *wgate;
        const uint8_t *wup;
        const uint8_t *wdown;
    } *layers;
} llama_model_t;

typedef struct {
    int    n_layer;
    int    n_head_kv;
    int    head_dim;
    int    capacity;
    int    seq_len;
    float *k;
    float *v;
} kv_cache_t;

int  llama_build_model(llama_model_t *m, gguf_model_t *params, gguf_tensor_info_t *tensors,
                       uint64_t n_tensors);
void llama_free_model(llama_model_t *m);
int  llama_scratch_init(llama_scratch_t *s, const llama_model_t *m);
void llama_scratch_free(llama_scratch_t *s);
int  kv_cache_init(kv_cache_t *cache, int n_layer, int n_head_kv, int head_dim, int capacity);
void kv_cache_free(kv_cache_t *cache);
void llama_forward(llama_model_t *m, kv_cache_t *cache, llama_scratch_t *scratch, int token,
                   int pos, float *logits);
int  sample(float *logits, int n_vocab, float temp, int top_k);
#endif /* INFER_H */
