#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>

#if defined(CONFIG_IDF_TARGET_ESP32S3)
#define EMBEDLM_PLATFORM_ESP32S3 1
#elif defined(STM32H7)
#define EMBEDLM_PLATFORM_STM32H7 1
#else
#define EMBEDLM_PLATFORM_HOST 1
#endif

#if EMBEDLM_PLATFORM_HOST
#include <time.h>
#endif

#if EMBEDLM_PLATFORM_STM32H7
#include "stm32h7xx_hal.h"
#endif

#if EMBEDLM_PLATFORM_ESP32S3
#include <esp_task_wdt.h>
#include <esp_timer.h>
#endif

#if EMBEDLM_PLATFORM_STM32H7 && !defined(EMBEDLM_SDCARD)
extern const uint8_t __embedlm_model_start[] __attribute__((weak));
extern const uint8_t __embedlm_model_end[] __attribute__((weak));
#endif

static inline void platform_init(void)
{
#if EMBEDLM_PLATFORM_STM32H7
    HAL_Init();
#endif
}

static inline void platform_watchdog_init(void)
{
#if EMBEDLM_PLATFORM_ESP32S3
    esp_task_wdt_config_t twdt = {.timeout_ms = 120000, .trigger_panic = false};
    esp_task_wdt_reconfigure(&twdt);
    esp_task_wdt_add(NULL);
#endif
}

static inline void platform_watchdog_kick(void)
{
#if EMBEDLM_PLATFORM_ESP32S3
    esp_task_wdt_reset();
#endif
}

static inline uint64_t platform_now_us(void)
{
#if EMBEDLM_PLATFORM_ESP32S3
    return esp_timer_get_time();
#elif EMBEDLM_PLATFORM_STM32H7
    return (uint64_t)HAL_GetTick() * 1000ULL;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
#endif
}

static inline const char *platform_default_model_path(void)
{
#if EMBEDLM_PLATFORM_STM32H7
    return "model.gguf";
#else
    return "large.gguf";
#endif
}

static inline int platform_embedded_model_matches(const void *buf)
{
#if EMBEDLM_PLATFORM_STM32H7 && !defined(EMBEDLM_SDCARD)
    return __embedlm_model_start != NULL && __embedlm_model_end != NULL &&
           buf == (const void *)__embedlm_model_start;
#else
    (void)buf;
    return 0;
#endif
}

static inline const uint8_t *platform_embedded_model_start(void)
{
#if EMBEDLM_PLATFORM_STM32H7 && !defined(EMBEDLM_SDCARD)
    return __embedlm_model_start;
#else
    return NULL;
#endif
}

static inline const uint8_t *platform_embedded_model_end(void)
{
#if EMBEDLM_PLATFORM_STM32H7 && !defined(EMBEDLM_SDCARD)
    return __embedlm_model_end;
#else
    return NULL;
#endif
}

#endif /* PLATFORM_H */
