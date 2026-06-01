#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>

static int sort_cmp(tokenizer_t *ctx, int ia, int ib)
{
    int min = ctx->lengths[ia] < ctx->lengths[ib] ? ctx->lengths[ia] : ctx->lengths[ib];
    int cmp = memcmp(ctx->strings[ia], ctx->strings[ib], min);
    if (cmp) return cmp;
    return ctx->lengths[ia] - ctx->lengths[ib];
}

static void sort_swap(int *a, int *b)
{
    int t = *a;
    *a    = *b;
    *b    = t;
}

static void sort_quicksort(tokenizer_t *tok, int lo, int hi)
{
    if (lo >= hi) return;
    int pivot = tok->sorted[(lo + hi) / 2];
    int i = lo, j = hi;
    while (i <= j)
    {
        while (sort_cmp(tok, tok->sorted[i], pivot) < 0) i++;
        while (sort_cmp(tok, tok->sorted[j], pivot) > 0) j--;
        if (i <= j)
        {
            sort_swap(&tok->sorted[i], &tok->sorted[j]);
            i++;
            j--;
        }
    }
    if (lo < j) sort_quicksort(tok, lo, j);
    if (i < hi) sort_quicksort(tok, i, hi);
}

static void sort_vocab(tokenizer_t *tok)
{
    sort_quicksort(tok, 0, tok->vocab_size - 1);
}

static int find_token(tokenizer_t *tok, const uint8_t *target, int target_len)
{
    int lo = 0, hi = tok->vocab_size;
    while (lo < hi)
    {
        int mid = lo + (hi - lo) / 2;
        int id  = tok->sorted[mid];
        int min = target_len < tok->lengths[id] ? target_len : tok->lengths[id];
        int cmp = memcmp(target, tok->strings[id], min);
        if (cmp == 0) cmp = target_len - tok->lengths[id];
        if (cmp < 0) hi = mid;
        else if (cmp > 0)
            lo = mid + 1;
        else
            return id;
    }
    return -1;
}

int tokenizer_init(tokenizer_t *tok)
{
    if (tok->bos_id < 0) tok->bos_id = 1;
    if (tok->eos_id < 0) tok->eos_id = 2;
    if (tok->unk_id < 0) tok->unk_id = 0;

    tok->max_token_len = 0;
    for (int i = 0; i < tok->vocab_size; i++)
        if (tok->lengths[i] > tok->max_token_len) tok->max_token_len = tok->lengths[i];

    tok->sorted = malloc(tok->vocab_size * sizeof(int));
    if (!tok->sorted) return -1;
    for (int i = 0; i < tok->vocab_size; i++) tok->sorted[i] = i;

    sort_vocab(tok);

    for (int i = 0; i < 256; i++) tok->byte_token[i] = -1;
    for (int i = 0; i < tok->vocab_size; i++)
        if (tok->lengths[i] == 1) tok->byte_token[(unsigned char)tok->strings[i][0]] = i;

    return 0;
}

int tokenizer_encode(tokenizer_t *tok, const char *text, int *ids, int max_ids)
{
    uint8_t norm[4096];
    int     nnorm = 0;

    for (int i = 0; text[i] && nnorm < (int)sizeof(norm) - 3; i++)
    {
        if (text[i] == ' ')
        {
            norm[nnorm++] = 0xE2;
            norm[nnorm++] = 0x96;
            norm[nnorm++] = 0x81;
        }
        else
        {
            norm[nnorm++] = (uint8_t)text[i];
        }
    }

    int n_ids = 0;
    if (n_ids < max_ids) ids[n_ids++] = tok->bos_id;

    int pos = 0;
    while (pos < nnorm)
    {
        int best_id  = -1;
        int best_len = 0;
        int max_len  = nnorm - pos;
        if (max_len > tok->max_token_len) max_len = tok->max_token_len;

        for (int len = max_len; len >= 1; len--)
        {
            int id = find_token(tok, norm + pos, len);
            if (id >= 0)
            {
                best_id  = id;
                best_len = len;
                break;
            }
        }

        if (best_id >= 0)
        {
            if (n_ids < max_ids) ids[n_ids++] = best_id;
            pos += best_len;
        }
        else
        {
            int bt = tok->byte_token[(unsigned char)norm[pos]];
            if (bt >= 0)
            {
                if (n_ids < max_ids) ids[n_ids++] = bt;
            }
            else if (tok->unk_id >= 0)
            {
                if (n_ids < max_ids) ids[n_ids++] = tok->unk_id;
            }
            pos++;
        }
    }

    return n_ids;
}

int tokenizer_decode(tokenizer_t *tok, const int *ids, int n_ids, char *text, int max_len)
{
    int pos = 0;
    for (int i = 0; i < n_ids && pos < max_len - 1; i++)
    {
        int id = ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;
        int len = tok->lengths[id];
        if (pos + len >= max_len) len = max_len - pos - 1;
        memcpy(text + pos, tok->strings[id], len);
        pos += len;
    }
    text[pos] = 0;

    int w = 0;
    for (int r = 0; r < pos;)
    {
        if (r + 2 < pos && (unsigned char)text[r] == 0xE2 && (unsigned char)text[r + 1] == 0x96 &&
            (unsigned char)text[r + 2] == 0x81)
        {
            text[w++] = ' ';
            r += 3;
        }
        else
        {
            text[w++] = text[r++];
        }
    }
    text[w] = 0;

    return w;
}

void tokenizer_free(tokenizer_t *tok)
{
    free(tok->sorted);
    free((void *)tok->strings);
    free(tok->lengths);
    free(tok->scores);
    free(tok->types);
}
