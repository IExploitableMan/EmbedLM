# EmbedLM

Optimized GGUF inference engine for running quantized llama models directly on microcontrollers.

## Quick Start

### ESP32-S3

```bash
# Build and flash
pio run -e esp32-s3-devkitc-1 -t upload

# Upload the model
python ~/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32s3 \
  --port /dev/ttyACM1 \
  --baud 115200 \
  write_flash 0x400000 model.gguf

# Monitor
pio device monitor -b 115200
```

### STM32H7

```bash
# Build and flash
pio run -e stm32h743ii-devboard -t upload

# Monitor
pio device monitor -b 115200
```

To enable SD-card loading, uncomment this in `platformio.ini`:

```ini
build_flags =
    -DEMBEDLM_SDCARD
```

### Linux

```bash
cmake --build build && EMBEDLM_MODEL_PATH=model.gguf ./build/embedlm
```
