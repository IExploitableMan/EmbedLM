#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_partition.h"
#include "spi_flash_mmap.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

typedef struct
{
    uint8_t magic[4];
    uint16_t vocab_size;
    uint16_t max_pos;
    uint16_t hidden_size;
    uint16_t intermediate_size;
    uint16_t num_layers;
    uint16_t num_heads;
    uint16_t num_merges;
    uint16_t byte_to_token[256];
    uint8_t token_to_byte[256];
    uint32_t merges[];
} __attribute__((packed)) emlm_t;

typedef struct
{
    float scale;
    int8_t w[];
} __attribute__((packed)) quant_t;

typedef struct
{
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    const quant_t *q_w, *k_w, *v_w, *out_w;
    const float *out_b;
    const quant_t *fc_w, *proj_w;
    const float *fc_b, *proj_b;
} layer_t;

#define MAX_SEQ 128

int tokenize(const emlm_t *data, const char *text, uint16_t *tokens, int max_len)
{
    int len = strlen(text);
    if (len > max_len)
        return -1;

    for (int i = 0; i < len; i++)
        tokens[i] = data->byte_to_token[(uint8_t)text[i]];

    for (uint32_t m = 0; m < data->num_merges; m++)
    {
        uint16_t a = data->merges[m] & 0xFFFF;
        uint16_t b = data->merges[m] >> 16;
        uint16_t result = 256 + m;

        int new_len = 0;
        for (int i = 0; i < len; i++)
        {
            if (i + 1 < len && tokens[i] == a && tokens[i + 1] == b)
            {
                tokens[new_len++] = result;
                i++;
            }
            else
            {
                tokens[new_len++] = tokens[i];
            }
        }
        len = new_len;
    }
    return len;
}

static int expand_token(const emlm_t *data, uint16_t token, uint8_t *out, int max_len)
{
    if (token < 256)
    {
        if (max_len < 1)
            return -1;
        out[0] = data->token_to_byte[token];
        return 1;
    }
    uint32_t merge = data->merges[token - 256];
    uint16_t a = merge & 0xFFFF;
    uint16_t b = merge >> 16;
    int la = expand_token(data, a, out, max_len);
    if (la < 0)
        return -1;
    int lb = expand_token(data, b, out + la, max_len - la);
    if (lb < 0)
        return -1;
    return la + lb;
}

int detokenize(const emlm_t *data, const uint16_t *tokens, int num_tokens, char *out, int max_len)
{
    int pos = 0;
    for (int i = 0; i < num_tokens; i++)
    {
        int n = expand_token(data, tokens[i], (uint8_t *)out + pos, max_len - pos);
        if (n < 0)
            return -1;
        pos += n;
    }
    out[pos] = '\0';
    return pos;
}

void embedding_lookup(const emlm_t *data, const quant_t *wte, const quant_t *wpe, int token_id, int pos, float *out)
{
    const int8_t *tok_row = wte->w + token_id * data->hidden_size;
    const int8_t *pos_row = wpe->w + pos * data->hidden_size;

    for (int i = 0; i < data->hidden_size; i++)
        out[i] = tok_row[i] * wte->scale + pos_row[i] * wpe->scale;
}

void layer_norm(const float *x, const float *weight, const float *bias, float *out, int n, float eps)
{
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++)
        mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++)
        var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float scale = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++)
        out[i] = (x[i] - mean) * scale * weight[i] + bias[i];
}

void matmul_q(const float *x, const quant_t *W, float *out, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        float sum = 0;
        for (int j = 0; j < cols; j++)
            sum += x[j] * W->w[i * cols + j];
        out[i] = sum * W->scale;
    }
}

float gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void mlp(const emlm_t *data, const float *x, const quant_t *fc_w, const float *fc_b, const quant_t *proj_w, const float *proj_b, float *out, int hidden, int intermediate)
{
    float fc[data->intermediate_size];
    matmul_q(x, fc_w, fc, intermediate, hidden);
    for (int i = 0; i < intermediate; i++)
        fc[i] = gelu(fc[i] + fc_b[i]);

    matmul_q(fc, proj_w, out, hidden, intermediate);
    for (int i = 0; i < hidden; i++)
        out[i] += proj_b[i];
}

void attention_kv(const float *x, const quant_t *q_w, const quant_t *k_w,
                  const quant_t *v_w, const quant_t *out_w, const float *out_b,
                  float *out, int hidden, int num_heads,
                  float *k_cache, float *v_cache, int pos)
{
    int head_dim = hidden / num_heads;
    float q[hidden], kv[hidden];

    matmul_q(x, q_w, q, hidden, hidden);

    matmul_q(x, k_w, kv, hidden, hidden);
    memcpy(k_cache + pos * hidden, kv, hidden * sizeof(float));

    matmul_q(x, v_w, kv, hidden, hidden);
    memcpy(v_cache + pos * hidden, kv, hidden * sizeof(float));

    float attn_out[hidden];
    memset(attn_out, 0, hidden * sizeof(float));

    for (int h = 0; h < num_heads; h++)
    {
        float *qh = q + h * head_dim;
        float scores[pos + 1];
        float max_s = -1e9f;

        for (int t = 0; t <= pos; t++)
        {
            float *kh = k_cache + t * hidden + h * head_dim;
            float s = 0;
            for (int i = 0; i < head_dim; i++)
                s += qh[i] * kh[i];
            s /= sqrtf((float)head_dim);
            scores[t] = s;
            if (s > max_s)
                max_s = s;
        }

        float sum = 0;
        for (int t = 0; t <= pos; t++)
        {
            scores[t] = expf(scores[t] - max_s);
            sum += scores[t];
        }
        for (int t = 0; t <= pos; t++)
            scores[t] /= sum;

        for (int i = 0; i < head_dim; i++)
        {
            float val = 0;
            for (int t = 0; t <= pos; t++)
                val += scores[t] * v_cache[t * hidden + h * head_dim + i];
            attn_out[h * head_dim + i] = val;
        }
    }

    matmul_q(attn_out, out_w, out, hidden, hidden);
    for (int i = 0; i < hidden; i++)
        out[i] += out_b[i];
}

int sample_token(float *logits, int vocab_size, float temperature, int top_k)
{
    if (top_k > vocab_size)
        top_k = vocab_size;

    for (int i = 0; i < vocab_size; i++)
        logits[i] /= temperature;

    int idx[top_k];
    float val[top_k];
    for (int j = 0; j < top_k; j++)
    {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best])
                best = i;
        idx[j] = best;
        val[j] = logits[best];
        logits[best] = -INFINITY;
    }

    float max_v = val[0];
    float sum = 0;
    for (int j = 0; j < top_k; j++)
    {
        val[j] = expf(val[j] - max_v);
        sum += val[j];
    }

    float r = (float)esp_random() / 4294967295.0f;
    float cum = 0;
    for (int j = 0; j < top_k; j++)
    {
        cum += val[j] / sum;
        if (r < cum)
            return idx[j];
    }
    return idx[0];
}

void parse_weights(const emlm_t *data, const quant_t **wte, const quant_t **wpe,
                   layer_t *layers, const float **lnf_w, const float **lnf_b)
{
    int h = data->hidden_size;
    int inter = data->intermediate_size;

    *wte = (const quant_t *)(data->merges + data->num_merges);
    *wpe = (const quant_t *)((uint8_t *)*wte + sizeof(float) + data->vocab_size * h);

    const uint8_t *ptr = (const uint8_t *)*wpe + sizeof(float) + data->max_pos * h;

    for (int l = 0; l < data->num_layers; l++)
    {
        layers[l].ln1_w = (const float *)ptr;
        ptr += h * sizeof(float);
        layers[l].ln1_b = (const float *)ptr;
        ptr += h * sizeof(float);
        layers[l].ln2_w = (const float *)ptr;
        ptr += h * sizeof(float);
        layers[l].ln2_b = (const float *)ptr;
        ptr += h * sizeof(float);

        layers[l].q_w = (const quant_t *)ptr;
        ptr += sizeof(float) + h * h;
        layers[l].k_w = (const quant_t *)ptr;
        ptr += sizeof(float) + h * h;
        layers[l].v_w = (const quant_t *)ptr;
        ptr += sizeof(float) + h * h;
        layers[l].out_w = (const quant_t *)ptr;
        ptr += sizeof(float) + h * h;
        layers[l].out_b = (const float *)ptr;
        ptr += h * sizeof(float);

        layers[l].fc_w = (const quant_t *)ptr;
        ptr += sizeof(float) + inter * h;
        layers[l].proj_w = (const quant_t *)ptr;
        ptr += sizeof(float) + h * inter;
        layers[l].fc_b = (const float *)ptr;
        ptr += inter * sizeof(float);
        layers[l].proj_b = (const float *)ptr;
        ptr += h * sizeof(float);
    }

    *lnf_w = (const float *)ptr;
    ptr += h * sizeof(float);
    *lnf_b = (const float *)ptr;
}

void generate(const emlm_t *data, const char *prompt, int max_tokens, float temperature, int top_k)
{
    int h = data->hidden_size;
    int inter = data->intermediate_size;
    int nl = data->num_layers;
    int nh = data->num_heads;
    int max_seq = MAX_SEQ;
    if (max_seq > data->max_pos)
        max_seq = data->max_pos;

    const quant_t *wte, *wpe;
    layer_t layers[nl];
    const float *lnf_w, *lnf_b;
    parse_weights(data, &wte, &wpe, layers, &lnf_w, &lnf_b);

    uint16_t tokens[max_seq];
    int prompt_len = tokenize(data, prompt, tokens, max_seq);
    if (prompt_len < 1)
    {
        printf("tokenization failed\n");
        return;
    }

    int cache_size = max_seq * h;
    float *k_cache = calloc(nl * cache_size, sizeof(float));
    float *v_cache = calloc(nl * cache_size, sizeof(float));
    float *logits = malloc(data->vocab_size * sizeof(float));
    if (!k_cache || !v_cache || !logits)
    {
        printf("alloc failed (need %d KB for kv cache)\n",
               (int)(nl * cache_size * sizeof(float) * 2 / 1024));
        free(k_cache);
        free(v_cache);
        free(logits);
        return;
    }

    printf("%s", prompt);
    fflush(stdout);

    int len = prompt_len;
    int generated = 0;
    float x[h];

    for (int pos = 0; pos < len && pos < max_seq - 1; pos++)
    {
        embedding_lookup(data, wte, wpe, tokens[pos], pos, x);

        for (int l = 0; l < nl; l++)
        {
            float *lk = k_cache + l * cache_size;
            float *lv = v_cache + l * cache_size;

            float tmp[h], out[h];

            layer_norm(x, layers[l].ln1_w, layers[l].ln1_b, tmp, h, 1e-5f);
            attention_kv(tmp, layers[l].q_w, layers[l].k_w, layers[l].v_w,
                         layers[l].out_w, layers[l].out_b, out, h, nh, lk, lv, pos);
            for (int i = 0; i < h; i++)
                x[i] += out[i];

            layer_norm(x, layers[l].ln2_w, layers[l].ln2_b, tmp, h, 1e-5f);
            mlp(data, tmp, layers[l].fc_w, layers[l].fc_b,
                layers[l].proj_w, layers[l].proj_b, out, h, inter);
            for (int i = 0; i < h; i++)
                x[i] += out[i];

            vTaskDelay(1);
        }

        if (pos >= prompt_len - 1)
        {
            layer_norm(x, lnf_w, lnf_b, x, h, 1e-5f);
            matmul_q(x, wte, logits, data->vocab_size, h);

            int next = sample_token(logits, data->vocab_size, temperature, top_k);

            char piece[64];
            detokenize(data, (uint16_t[]){(uint16_t)next}, 1, piece, sizeof(piece));
            printf("%s", piece);
            fflush(stdout);

            tokens[len++] = next;
            generated++;
            if (generated >= max_tokens)
                break;

            vTaskDelay(1);
        }
    }

    printf("\n[%d tokens generated]\n", generated);
    free(k_cache);
    free(v_cache);
    free(logits);
}

void app_main(void)
{
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "emlm");
    if (!part)
    {
        printf("Partition not found\n");
        return;
    }

    esp_partition_mmap_handle_t mmap_handle;
    const void *mapped;
    esp_err_t err = esp_partition_mmap(
        part, 0, part->size, SPI_FLASH_MMAP_DATA, &mapped, &mmap_handle);
    if (err != ESP_OK)
    {
        printf("Failed to mmap partition: %d\n", err);
        return;
    }

    const emlm_t *data = (const emlm_t *)mapped;
    printf("EMLM: %c%c%c%c vocab=%u hidden=%u layers=%u heads=%u\n",
           data->magic[0], data->magic[1], data->magic[2], data->magic[3],
           data->vocab_size, data->hidden_size, data->num_layers, data->num_heads);
    printf("KV cache: %d KB\n",
           (int)(2 * data->num_layers * MAX_SEQ * data->hidden_size * sizeof(float) / 1024));

    generate(data, "Once upon a time", 100, 0.5f, 40);
}
