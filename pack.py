import struct
import numpy as np

bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1

u2b = {chr(c): b for b, c in zip(bs, cs)}
ids = {bytes([b]): i for i, b in enumerate(bs)}

byte_to_token = [0] * 256
token_to_byte = [0] * 256
for i, b in enumerate(bs):
    byte_to_token[b] = i
    token_to_byte[i] = b

merges = []
for line in open("merges.txt", encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#version:"):
        continue
    a, b = line.rsplit(" ", 1)
    a = bytes(u2b[c] for c in a)
    b = bytes(u2b[c] for c in b)
    merges.append((ids[a], ids[b]))
    ids[a + b] = 256 + len(merges) - 1

from transformers import AutoModelForCausalLM

ID = "roneneldan/TinyStories-1M"
model = AutoModelForCausalLM.from_pretrained(ID)
sd = model.state_dict()


def q(key):
    t = sd[key].float().numpy()
    scale = max(np.abs(t).max() / 127.0, 1e-8)
    return np.clip(np.round(t / scale), -128, 127).astype(np.int8), scale


def f32(key):
    return sd[key].float().numpy().flatten().tobytes()


with open("emlm.bin", "wb") as f:
    f.write(b"EMLM")
    f.write(struct.pack("<H", 50257))  # vocab size
    f.write(struct.pack("<H", 2048))   # max position
    f.write(struct.pack("<H", 64))     # hidden size
    f.write(struct.pack("<H", 256))    # intermediate size
    f.write(struct.pack("<H", 8))      # num layers
    f.write(struct.pack("<H", 16))     # num heads
    f.write(struct.pack("<H", len(merges)))  # num merges

    for t in byte_to_token:
        f.write(struct.pack("<H", t))
    for t in token_to_byte:
        f.write(struct.pack("<B", t))
    for a, b in merges:
        f.write(struct.pack("<HH", a, b))

    qi, s = q("transformer.wte.weight")
    f.write(struct.pack("<f", s))
    f.write(qi.tobytes())
    qi, s = q("transformer.wpe.weight")
    f.write(struct.pack("<f", s))
    f.write(qi.tobytes())

    for i in range(8):
        p = f"transformer.h.{i}"
        f.write(f32(f"{p}.ln_1.weight"))
        f.write(f32(f"{p}.ln_1.bias"))
        f.write(f32(f"{p}.ln_2.weight"))
        f.write(f32(f"{p}.ln_2.bias"))
        for w in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            qi, s = q(f"{p}.attn.attention.{w}.weight")
            f.write(struct.pack("<f", s))
            f.write(qi.tobytes())
        f.write(f32(f"{p}.attn.attention.out_proj.bias"))
        for w in ["c_fc", "c_proj"]:
            qi, s = q(f"{p}.mlp.{w}.weight")
            f.write(struct.pack("<f", s))
            f.write(qi.tobytes())
        f.write(f32(f"{p}.mlp.c_fc.bias"))
        f.write(f32(f"{p}.mlp.c_proj.bias"))

    f.write(f32("transformer.ln_f.weight"))
    f.write(f32("transformer.ln_f.bias"))

import os

print(f"emlm.bin: {os.path.getsize("emlm.bin") / 1024 / 1024:.2f} MB")
