#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t  byte_to_token[256];

static uint32_t *merges;
static uint32_t  num_merges;

static int load_merges(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    if (fread(byte_to_token, 2, 256, f) != 256) { fclose(f); return -1; }

    uint16_t count;
    if (fread(&count, 2, 1, f) != 1) { fclose(f); return -1; }
    num_merges = count;

    merges = malloc(num_merges * sizeof(uint32_t));
    if (!merges) { fclose(f); return -1; }

    for (uint32_t i = 0; i < num_merges; i++) {
        uint16_t a, b;
        if (fread(&a, 2, 1, f) != 1 || fread(&b, 2, 1, f) != 1) {
            fclose(f); free(merges); return -1;
        }
        merges[i] = ((uint32_t)b << 16) | a;
    }

    fclose(f);
    return 0;
}

/* Returns number of tokens, or -1 on error.
   tokens[] is allocated by the caller (max_len elements). */
static int tokenize(const char *text, size_t text_len,
                    uint16_t *tokens, size_t max_len)
{
    if (text_len > max_len) return -1;

    int len = (int)text_len;
    for (int i = 0; i < len; i++)
        tokens[i] = byte_to_token[(uint8_t)text[i]];

    for (uint32_t m = 0; m < num_merges; m++) {
        uint16_t a      = (uint16_t)(merges[m] & 0xFFFF);
        uint16_t b      = (uint16_t)(merges[m] >> 16);
        uint16_t result = (uint16_t)(256 + m);

        int new_len = 0;
        for (int i = 0; i < len; i++) {
            if (i + 1 < len && tokens[i] == a && tokens[i + 1] == b) {
                tokens[new_len++] = result;
                i++;
            } else {
                tokens[new_len++] = tokens[i];
            }
        }
        len = new_len;
    }
    return len;
}

int main(void)
{
    if (load_merges("merges.bin") != 0)
        return 1;

    const char *text = "Hello, how are you?";
    size_t text_len = strlen(text);

    uint16_t *tokens = malloc(text_len * sizeof(uint16_t));
    if (!tokens) { free(merges); return 1; }

    int n = tokenize(text, text_len, tokens, text_len);
    if (n < 0) {
        puts("tokenize error");
    } else {
        printf("%d tokens:", n);
        for (int i = 0; i < n; i++)
            printf(" %u", tokens[i]);
        putchar('\n');
    }

    free(tokens);
    free(merges);
    return 0;
}