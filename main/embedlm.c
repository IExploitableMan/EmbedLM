#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_partition.h"
#include "spi_flash_mmap.h"

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
    uint32_t merges[]; /* num_merges entries, then model weights follow */
} __attribute__((packed)) emlm_t;

typedef struct
{
    float scale;
    int8_t w[];
} __attribute__((packed)) quant_t;

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

void attention(const emlm_t *data, const float *x, const quant_t *q_w, const quant_t *k_w, const quant_t *v_w, const quant_t *out_w, const float *out_b, float *out, int hidden, int num_heads)
{
    int head_dim = hidden / num_heads;
    float q[data->hidden_size], k[data->hidden_size], v[data->hidden_size];

    matmul_q(x, q_w, q, hidden, hidden);
    matmul_q(x, k_w, k, hidden, hidden);
    matmul_q(x, v_w, v, hidden, hidden);

    float attn_out[data->hidden_size];
    memset(attn_out, 0, sizeof(attn_out));

    for (int h = 0; h < num_heads; h++)
    {
        float *qh = q + h * head_dim;
        float *kh = k + h * head_dim;
        float *vh = v + h * head_dim;

        float score = 0;
        for (int i = 0; i < head_dim; i++)
            score += qh[i] * kh[i];
        score /= sqrtf((float)head_dim);

        score = 1.0f; // FIX: softmax

        for (int i = 0; i < head_dim; i++)
            attn_out[h * head_dim + i] = score * vh[i];
    }

    matmul_q(attn_out, out_w, out, hidden, hidden);
    for (int i = 0; i < hidden; i++)
        out[i] += out_b[i];
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
        part,
        0,
        part->size,
        SPI_FLASH_MMAP_DATA,
        &mapped,
        &mmap_handle);
    if (err != ESP_OK)
    {
        printf("Failed to mmap partition: %d\n", err);
        return;
    }

    const emlm_t *data = (const emlm_t *)mapped;

    printf("Magic: %c%c%c%c\n", data->magic[0], data->magic[1], data->magic[2], data->magic[3]);
    printf("Vocab size: %u\n", data->vocab_size);
    printf("Hidden size: %u\n", data->hidden_size);
    printf("Number of layers: %u\n", data->num_layers);
    printf("Number of heads: %u\n", data->num_heads);

    uint16_t tokens[128];
    int n = tokenize(data, "Hello, how are you?", tokens, 128);

    const quant_t *wte = (const quant_t *)(data->merges + data->num_merges);
    const quant_t *wpe = (const quant_t *)((uint8_t *)wte + sizeof(float) + data->vocab_size * data->hidden_size);
    float embedding[data->hidden_size];
    embedding_lookup(data, wte, wpe, tokens[0], 0, embedding);

    const uint8_t *ptr = (const uint8_t *)wpe + sizeof(float) + data->max_pos * data->hidden_size;
    float x[data->hidden_size];
    memcpy(x, embedding, sizeof(x));
    for (int layer = 0; layer < data->num_layers; layer++)
    {
        const float *ln1_w = (const float *)ptr;
        ptr += data->hidden_size * sizeof(float);
        const float *ln1_b = (const float *)ptr;
        ptr += data->hidden_size * sizeof(float);
        const float *ln2_w = (const float *)ptr;
        ptr += data->hidden_size * sizeof(float);
        const float *ln2_b = (const float *)ptr;
        ptr += data->hidden_size * sizeof(float);
        float ln_out[data->hidden_size];
        layer_norm(x, ln1_w, ln1_b, ln_out, data->hidden_size, 1e-5);

        const quant_t *q_w = (const quant_t *)ptr;
        ptr += sizeof(float) + data->hidden_size * data->hidden_size;
        const quant_t *k_w = (const quant_t *)ptr;
        ptr += sizeof(float) + data->hidden_size * data->hidden_size;
        const quant_t *v_w = (const quant_t *)ptr;
        ptr += sizeof(float) + data->hidden_size * data->hidden_size;
        const quant_t *out_w = (const quant_t *)ptr;
        ptr += sizeof(float) + data->hidden_size * data->hidden_size;
        const float *out_b = (const float *)ptr;
        ptr += data->hidden_size * sizeof(float);
        float attn_out[data->hidden_size];
        attention(data, ln_out, q_w, k_w, v_w, out_w, out_b, attn_out, data->hidden_size, data->num_heads);

        float resid1[data->hidden_size];
        for (int i = 0; i < data->hidden_size; i++)
            resid1[i] = x[i] + attn_out[i];

        float ln2_out[data->hidden_size];
        layer_norm(resid1, ln2_w, ln2_b, ln2_out, data->hidden_size, 1e-5f);

        const quant_t *fc_w = (const quant_t *)ptr;
        ptr += sizeof(float) + data->intermediate_size * data->hidden_size;
        const quant_t *proj_w = (const quant_t *)ptr;
        ptr += sizeof(float) + data->hidden_size * data->intermediate_size;
        const float *fc_b = (const float *)ptr;
        ptr += data->intermediate_size * sizeof(float);
        const float *proj_b = (const float *)ptr;
        ptr += data->hidden_size * sizeof(float);
        float mlp_out[data->hidden_size];
        mlp(data, ln2_out, fc_w, fc_b, proj_w, proj_b, mlp_out, data->hidden_size, data->intermediate_size);

        for (int i = 0; i < data->hidden_size; i++)
            x[i] = resid1[i] + mlp_out[i];
        printf("layer %d output: % .4f % .4f % .4f % .4f\n", layer, x[0], x[1], x[2], x[3]);
    }

    const float *lnf_w = (const float *)ptr;
    ptr += data->hidden_size * sizeof(float);
    const float *lnf_b = (const float *)ptr;
    ptr += data->hidden_size * sizeof(float);
    float lnf_out[data->hidden_size];
    layer_norm(x, lnf_w, lnf_b, lnf_out, data->hidden_size, 1e-5f);
    printf("final output: % .4f % .4f % .4f % .4f\n", lnf_out[0], lnf_out[1], lnf_out[2], lnf_out[3]);

    float *logits = (float *)malloc(data->vocab_size * sizeof(float));
    matmul_q(lnf_out, wte, logits, data->vocab_size, data->hidden_size);
    int best = 0;
    for (int i = 1; i < data->vocab_size; i++)
        if (logits[i] > logits[best])
            best = i;
    free(logits);
    detokenize(data, (uint16_t[]){best}, 1, (char *)lnf_out, sizeof(lnf_out));
    printf("predicted next token: %s\n", (char *)lnf_out);
}
