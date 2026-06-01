#include "infer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define Q8_BLOCK_SIZE 34
#define Q8_N 32

static inline float fp16_to_fp32(uint16_t h)
{
    uint32_t sign   = ((uint32_t)h >> 15) << 31;
    uint32_t exp5   = (h >> 10) & 0x1f;
    uint32_t mant10 = h & 0x3ff;
    uint32_t exp8, mant23;

    if (exp5 == 0x1f)
    {
        exp8   = 0xff;
        mant23 = mant10 << 13;
    }
    else if (exp5 == 0)
    {
        if (mant10 == 0)
        {
            exp8   = 0;
            mant23 = 0;
        }
        else
        {
            int exp = -14;
            while ((mant10 & 0x400) == 0)
            {
                mant10 <<= 1;
                exp--;
            }
            exp8   = (uint32_t)(exp + 127);
            mant23 = (mant10 & 0x3ff) << 13;
        }
    }
    else
    {
        exp8   = exp5 + 112;
        mant23 = mant10 << 13;
    }

    uint32_t f = sign | (exp8 << 23) | mant23;
    float    r;
    memcpy(&r, &f, 4);
    return r;
}

static void q8_matmul(const float *restrict x, const uint8_t *restrict w, float *restrict y,
                      int n_rows, int n_cols)
{
    int n_blocks = (n_rows + Q8_N - 1) / Q8_N;

    for (int c = 0; c < n_cols; c++)
    {
        const uint8_t *col = w + (size_t)c * n_blocks * Q8_BLOCK_SIZE;
        float          sum = 0;
        for (int b = 0; b < n_blocks; b++)
        {
            uint16_t dbits;
            memcpy(&dbits, col + (size_t)b * Q8_BLOCK_SIZE, 2);
            float         d   = fp16_to_fp32(dbits);
            const int8_t *qs  = (const int8_t *)(col + (size_t)b * Q8_BLOCK_SIZE + 2);
            int           rem = n_rows - b * Q8_N;
            int           n   = rem < Q8_N ? rem : Q8_N;
            float         dot = 0;
            for (int k = 0; k < n; k++) dot += x[b * Q8_N + k] * qs[k];
            sum += d * dot;
        }
        y[c] = sum;
    }
}

static void rms_norm(const float *restrict x, const float *restrict weight, float *restrict y,
                     int n, float eps)
{
    float ss = 0, comp = 0;
    for (int i = 0; i < n; i++)
    {
        float val  = x[i] * x[i] - comp;
        float next = ss + val;
        comp       = (next - ss) - val;
        ss         = next;
    }
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * scale * weight[i];
}

static void rope(float *q, float *k, int n_head, int n_head_kv, int head_dim, int pos,
                 const float *inv_freq)
{
    for (int d = 0; d < head_dim; d += 2)
    {
        float freq = inv_freq[d / 2];
        float th   = pos * freq;
        float c    = cosf(th);
        float s    = sinf(th);

        for (int h = 0; h < n_head; h++)
        {
            int   off      = h * head_dim;
            float q0       = q[off + d];
            float q1       = q[off + d + 1];
            q[off + d]     = q0 * c - q1 * s;
            q[off + d + 1] = q0 * s + q1 * c;
        }
        for (int h = 0; h < n_head_kv; h++)
        {
            int   off      = h * head_dim;
            float kv0      = k[off + d];
            float kv1      = k[off + d + 1];
            k[off + d]     = kv0 * c - kv1 * s;
            k[off + d + 1] = kv0 * s + kv1 * c;
        }
    }
}

static inline float silu(float x)
{
    return x / (1.0f + expf(-x));
}

static void attention_layer(const float *q, kv_cache_t *cache, int layer, float *out, int n_head,
                            int head_dim, float *score)
{
    int   n_head_kv = cache->n_head_kv;
    int   group     = n_head / n_head_kv;
    int   seq_len   = cache->seq_len;
    float scale     = 1.0f / sqrtf((float)head_dim);
    int   stride    = n_head_kv * head_dim;
    int   layer_off = cache->capacity * stride;

    if (seq_len <= 0) return;

    for (int h = 0; h < n_head; h++)
    {
        int          kvh = h / group;
        const float *qh  = q + h * head_dim;

        float maxs = -INFINITY;
        for (int p = 0; p < seq_len; p++)
        {
            const float *kp =
                cache->k + (size_t)layer * layer_off + (size_t)p * stride + (size_t)kvh * head_dim;
            float s = 0;
            for (int d = 0; d < head_dim; d++) s += qh[d] * kp[d];
            s *= scale;
            score[p] = s;
            if (s > maxs) maxs = s;
        }

        float sum = 0;
        for (int p = 0; p < seq_len; p++)
        {
            score[p] = expf(score[p] - maxs);
            sum += score[p];
        }
        float isum = 1.0f / sum;
        for (int p = 0; p < seq_len; p++) score[p] *= isum;

        float *oh = out + h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        for (int p = 0; p < seq_len; p++)
        {
            const float *vp =
                cache->v + (size_t)layer * layer_off + (size_t)p * stride + (size_t)kvh * head_dim;
            float sp = score[p];
            for (int d = 0; d < head_dim; d++) oh[d] += sp * vp[d];
        }
    }
}

int llama_build_model(llama_model_t *m, gguf_model_t *params, gguf_tensor_info_t *tensors,
                      uint64_t n_tensors)
{
    memset(m, 0, sizeof(*m));
    m->n_layer        = (int)params->n_layer;
    m->n_embd         = (int)params->n_embd;
    m->n_head         = (int)params->n_head;
    m->n_head_kv      = (int)params->n_head_kv;
    m->n_ff           = (int)params->n_ff;
    m->n_vocab        = (int)params->n_vocab;
    m->rope_freq_base = params->rope_freq_base > 0 ? params->rope_freq_base : 10000.0f;
    m->rms_norm_eps   = params->rms_norm_eps > 0 ? params->rms_norm_eps : 1e-5f;
    if (m->n_vocab <= 0)
    {
        printf("error: invalid vocab size\n");
        return -1;
    }

    if (m->n_embd <= 0 || m->n_head <= 0 || m->n_head_kv <= 0)
    {
        printf("error: invalid model dimensions\n");
        return -1;
    }

    m->output_weight = NULL;

    if (m->n_embd % m->n_head != 0)
    {
        printf("error: n_embd not divisible by n_head\n");
        return -1;
    }

    m->head_dim = m->n_embd / m->n_head;

    int half_dim = m->head_dim / 2;
    m->inv_freq  = malloc(half_dim * sizeof(float));
    if (!m->inv_freq) return -1;
    for (int i = 0; i < half_dim; i++)
        m->inv_freq[i] = powf(m->rope_freq_base, -2.0f * (float)i / (float)m->head_dim);

    m->layers = malloc(m->n_layer * sizeof(*m->layers));
    if (!m->layers)
    {
        free(m->inv_freq);
        m->inv_freq = NULL;
        return -1;
    }
    memset(m->layers, 0, m->n_layer * sizeof(*m->layers));

    for (uint64_t i = 0; i < n_tensors; i++)
    {
        const char *name = tensors[i].name.data;
        size_t      nlen = tensors[i].name.len;

#define TMATCH(s) (nlen == sizeof(s) - 1 && memcmp(name, s, sizeof(s) - 1) == 0)

        if (TMATCH("output_norm.weight")) m->output_norm = (const float *)tensors[i].data;
        else if (TMATCH("output.weight"))
            m->output_weight = tensors[i].data;
        else if (TMATCH("token_embd.weight"))
            m->token_embd = tensors[i].data;
        else
        {
            for (int l = 0; l < m->n_layer; l++)
            {
                char buf[64];
                int  n = snprintf(buf, sizeof(buf), "blk.%d.attn_norm.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].attn_norm = (const float *)tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.attn_q.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].wq = tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.attn_k.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].wk = tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.attn_v.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].wv = tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.attn_output.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].wo = tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.ffn_norm.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].ffn_norm = (const float *)tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.ffn_gate.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].wgate = tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.ffn_up.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].wup = tensors[i].data;
                    break;
                }
                n = snprintf(buf, sizeof(buf), "blk.%d.ffn_down.weight", l);
                if ((int)nlen == n && memcmp(name, buf, n) == 0)
                {
                    m->layers[l].wdown = tensors[i].data;
                    break;
                }
            }
        }
    }

    if (!m->output_norm || !m->token_embd)
    {
        printf("error: missing output_norm or token_embd\n");
        free(m->layers);
        m->layers = NULL;
        free(m->inv_freq);
        m->inv_freq = NULL;
        return -1;
    }
    for (int l = 0; l < m->n_layer; l++)
    {
        if (!m->layers[l].attn_norm || !m->layers[l].wq || !m->layers[l].wk || !m->layers[l].wv ||
            !m->layers[l].wo || !m->layers[l].ffn_norm || !m->layers[l].wgate ||
            !m->layers[l].wup || !m->layers[l].wdown)
        {
            printf("error: missing layer %d tensor\n", l);
            free(m->layers);
            m->layers = NULL;
            free(m->inv_freq);
            m->inv_freq = NULL;
            return -1;
        }
    }

    return 0;
}

void llama_free_model(llama_model_t *m)
{
    free(m->inv_freq);
    free(m->layers);
    m->inv_freq = NULL;
    m->layers   = NULL;
}

int llama_scratch_init(llama_scratch_t *s, const llama_model_t *m)
{
    int buf_sz  = m->n_embd > m->n_ff ? m->n_embd : m->n_ff;
    int nk      = m->n_head_kv * m->head_dim;
    s->x        = malloc(m->n_embd * sizeof(float));
    s->buf      = malloc(buf_sz * sizeof(float));
    s->attn_out = malloc(m->n_embd * sizeof(float));
    s->q        = malloc(m->n_embd * sizeof(float));
    s->k        = malloc(nk * sizeof(float));
    s->v        = malloc(nk * sizeof(float));
    s->gate     = malloc(m->n_ff * sizeof(float));
    s->up       = malloc(m->n_ff * sizeof(float));
    s->score    = malloc(LLAMA_MAX_CTX * sizeof(float));
    s->probs    = malloc(m->n_vocab * sizeof(float));
    if (!s->x || !s->buf || !s->attn_out || !s->q || !s->k || !s->v || !s->gate || !s->up ||
        !s->score || !s->probs)
    {
        llama_scratch_free(s);
        return -1;
    }
    return 0;
}

void llama_scratch_free(llama_scratch_t *s)
{
    free(s->x);
    free(s->buf);
    free(s->attn_out);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->gate);
    free(s->up);
    free(s->score);
    free(s->probs);
    s->x = s->buf = s->attn_out = NULL;
    s->q = s->k = s->v = NULL;
    s->gate = s->up = s->score = s->probs = NULL;
}

int kv_cache_init(kv_cache_t *cache, int n_layer, int n_head_kv, int head_dim, int capacity)
{
    memset(cache, 0, sizeof(*cache));
    cache->n_layer   = n_layer;
    cache->n_head_kv = n_head_kv;
    cache->head_dim  = head_dim;
    cache->capacity  = capacity;
    cache->seq_len   = 0;

    int    stride = n_head_kv * head_dim;
    size_t sz     = (size_t)n_layer * (size_t)capacity * (size_t)stride;
    cache->k      = malloc(sz * sizeof(float));
    cache->v      = malloc(sz * sizeof(float));
    if (!cache->k || !cache->v)
    {
        free(cache->k);
        free(cache->v);
        return -1;
    }
    return 0;
}

void kv_cache_free(kv_cache_t *cache)
{
    free(cache->k);
    free(cache->v);
    memset(cache, 0, sizeof(*cache));
}

void llama_forward(llama_model_t *m, kv_cache_t *cache, llama_scratch_t *scratch, int token,
                   int pos, float *logits)
{
    int n_embd    = m->n_embd;
    int n_head    = m->n_head;
    int n_head_kv = m->n_head_kv;
    int head_dim  = m->head_dim;
    int n_ff      = m->n_ff;
    int n_vocab   = m->n_vocab;

    int nk = n_head_kv * head_dim;

    float *x        = scratch->x;
    float *buf      = scratch->buf;
    float *attn_out = scratch->attn_out;
    float *q        = scratch->q;
    float *k        = scratch->k;
    float *v        = scratch->v;
    float *gate     = scratch->gate;
    float *up       = scratch->up;

    memset(x, 0, n_embd * sizeof(float));
    int            n_blocks = (n_embd + Q8_N - 1) / Q8_N;
    const uint8_t *emb_data = m->token_embd + (size_t)token * n_blocks * Q8_BLOCK_SIZE;
    for (int b = 0; b < n_blocks; b++)
    {
        uint16_t dbits;
        memcpy(&dbits, emb_data + (size_t)b * Q8_BLOCK_SIZE, 2);
        float         d   = fp16_to_fp32(dbits);
        const int8_t *qs  = (const int8_t *)(emb_data + (size_t)b * Q8_BLOCK_SIZE + 2);
        int           rem = n_embd - b * Q8_N;
        int           n   = rem < Q8_N ? rem : Q8_N;
        for (int ki = 0; ki < n; ki++) x[b * Q8_N + ki] = qs[ki] * d;
    }

    int kv_stride    = n_head_kv * head_dim;
    int kv_layer_off = cache->capacity * kv_stride;

    for (int l = 0; l < m->n_layer; l++)
    {
        rms_norm(x, m->layers[l].attn_norm, buf, n_embd, m->rms_norm_eps);

        int nq = n_head * head_dim;
        q8_matmul(buf, m->layers[l].wq, q, n_embd, nq);
        q8_matmul(buf, m->layers[l].wk, k, n_embd, nk);
        q8_matmul(buf, m->layers[l].wv, v, n_embd, nk);

        rope(q, k, n_head, n_head_kv, head_dim, pos, m->inv_freq);

        int layer_off = l * kv_layer_off;
        memcpy(cache->k + layer_off + (size_t)pos * kv_stride, k, nk * sizeof(float));
        memcpy(cache->v + layer_off + (size_t)pos * kv_stride, v, nk * sizeof(float));

        if (pos + 1 > cache->seq_len) cache->seq_len = pos + 1;

        attention_layer(q, cache, l, buf, n_head, head_dim, scratch->score);

        q8_matmul(buf, m->layers[l].wo, attn_out, nq, n_embd);

        for (int i = 0; i < n_embd; i++) attn_out[i] += x[i];

        rms_norm(attn_out, m->layers[l].ffn_norm, buf, n_embd, m->rms_norm_eps);

        q8_matmul(buf, m->layers[l].wgate, gate, n_embd, n_ff);
        q8_matmul(buf, m->layers[l].wup, up, n_embd, n_ff);

        for (int i = 0; i < n_ff; i++) gate[i] = silu(gate[i]) * up[i];

        q8_matmul(gate, m->layers[l].wdown, buf, n_ff, n_embd);

        for (int i = 0; i < n_embd; i++) x[i] = buf[i] + attn_out[i];
    }

    rms_norm(x, m->output_norm, buf, n_embd, m->rms_norm_eps);

    const uint8_t *output_w = m->output_weight ? m->output_weight : m->token_embd;
    q8_matmul(buf, output_w, logits, n_embd, n_vocab);
}

int sample(float *logits, int n_vocab, float temp, int top_k, float *probs)
{
    if (temp <= 0.0f || top_k == 1)
    {
        int best = 0;
        for (int i = 1; i < n_vocab; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    if (!probs)
    {
        int best = 0;
        for (int i = 1; i < n_vocab; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < n_vocab; i++) probs[i] = logits[i] / temp;

    float mx = probs[0];
    for (int i = 1; i < n_vocab; i++)
        if (probs[i] > mx) mx = probs[i];
    float sum = 0;
    for (int i = 0; i < n_vocab; i++)
    {
        probs[i] = expf(probs[i] - mx);
        sum += probs[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n_vocab; i++) probs[i] *= inv;

    if (top_k > 0 && top_k < n_vocab)
    {
        float lo = 0, hi = 1.0f;
        for (int iter = 0; iter < 32; iter++)
        {
            float mid = (lo + hi) * 0.5f;
            int   cnt = 0;
            for (int i = 0; i < n_vocab; i++)
                if (probs[i] >= mid) cnt++;
            if (cnt >= top_k) lo = mid;
            else
                hi = mid;
        }
        sum = 0;
        for (int i = 0; i < n_vocab; i++)
        {
            if (probs[i] < lo) probs[i] = 0;
            sum += probs[i];
        }
        inv = 1.0f / sum;
        for (int i = 0; i < n_vocab; i++) probs[i] *= inv;
    }

    float r   = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    int   idx = n_vocab - 1;
    for (int i = 0; i < n_vocab; i++)
    {
        cum += probs[i];
        if (r < cum)
        {
            idx = i;
            break;
        }
    }
    return idx;
}
