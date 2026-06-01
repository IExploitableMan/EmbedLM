#if defined(EMBEDLM_PLATFORM_STM32) && defined(EMBEDLM_ENABLE_FLASH_MODEL)

    .section .rodata.embedlm_model,"a",%progbits
    .balign 4
    .global __embedlm_model_start
    .global __embedlm_model_end

__embedlm_model_start:
    .incbin "../model.gguf"
__embedlm_model_end:

#endif
