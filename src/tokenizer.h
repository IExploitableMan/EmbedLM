#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int          vocab_size;
    int          max_token_len;
    const char **strings;
    int         *lengths;
    float       *scores;
    int         *types;
    int         *sorted;
    int          byte_token[256];
    int          bos_id;
    int          eos_id;
    int          unk_id;
} tokenizer_t;

int  tokenizer_init(tokenizer_t *tok);
int  tokenizer_encode(tokenizer_t *tok, const char *text, int *ids, int max_ids);
int  tokenizer_decode(tokenizer_t *tok, const int *ids, int n_ids, char *text, int max_len);
void tokenizer_free(tokenizer_t *tok);

#endif /* TOKENIZER_H */
