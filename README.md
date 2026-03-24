# EmbedLM

Optimized inference engine for running quantized GPTNeo models directly on ESP32 microcontrollers. It features INT8 quantization and memory-mapped flash execution to enable LLMs on embedded hardware.

## Quick Start

```bash
# Prepare and quantize the model
python pack.py

# Flash the firmware and the model weights
idf.py build flash
parttool.py write_partition --partition-name emlm --input emlm.bin

# Monitor output
idf.py monitor
```
