# EmbedLM

Optimized inference engine for running quantized GPTNeo models directly on ESP32 microcontrollers. It features INT8 quantization and memory-mapped flash execution to enable LLMs on embedded hardware.

## Quick Start

```bash
# Create venv
python -m venv .venv

# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Prepare and quantize the model
python pack.py

# Flash the firmware
pio run -t upload

# Upload binary using crappy way
python ~/.platformio/packages/tool-esptoolpy/esptool.py --chip esp32s3 --port /dev/ttyACM1 --baud 115200 write_flash 0x400000 emlm.bin

# Monitor output
pio device monitor
```
