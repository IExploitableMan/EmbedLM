# EmbedLM

Optimized GGUF inference engine for running quantized llama models directly on microcontrollers.

## Quick Start

```bash
# Flash the firmware
pio run -t upload

# [ESP] Upload binary using crappy way
python ~/.platformio/packages/tool-esptoolpy/esptool.py --chip esp32s3 --port /dev/ttyACM1 --baud 115200 write_flash 0x400000 emlm.bin

# Monitor output
pio device monitor
```
