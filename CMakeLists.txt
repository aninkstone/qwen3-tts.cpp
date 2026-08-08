# 💡 修改 1：将最低版本提升至 3.18，以支持现代化的 find_package(CUDAToolkit)
cmake_minimum_required(VERSION 3.18)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static libs are linked into the qwen3tts shared lib (C API), so they must be PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
endif()

# Find threads
find_package(Threads REQUIRED)

# ========================================================================
# Inker Engine: 注入外部 GGML 依赖 & 硬件后端自动路由
# ========================================================================
set(INKER_GGML_INCLUDE_DIR "" CACHE PATH "Path to Inker's global ggml include directory")
set(INKER_GGML_LIB_DIR "" CACHE PATH "Path to Inker's global ggml library directory")

if(NOT INKER_GGML_INCLUDE_DIR OR NOT INKER_GGML_LIB_DIR)
    message(FATAL_ERROR "[Inker Engine] INKER_GGML_INCLUDE_DIR and INKER_GGML_LIB_DIR must be explicitly provided!")
endif()

# 1. 探测 Metal 后端 (macOS)
if(APPLE AND EXISTS "${INKER_GGML_LIB_DIR}/libggml-metal.dylib")
    set(GGML_HAS_METAL ON)
    message(STATUS "[Inker Engine] GGML Metal backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_METAL OFF)
endif()

# 2. 探测 CUDA 后端 (Windows) - 💡 修改 2：专为 cors.dll 满血版打通链路
if(EXISTS "${INKER_GGML_LIB_DIR}/ggml-cuda.lib")
    set(GGML_HAS_CUDA ON)
    message(STATUS "[Inker Engine] GGML CUDA backend found — Nvidia GPU acceleration enabled")
    find_package(CUDAToolkit REQUIRED)
else()
    set(GGML_HAS_CUDA OFF)
endif()

# 3. 封装统一的 GGML 链接宏 (保持代码极致整洁) - 💡 修改 3
macro(inker_link_ggml_backend target_name)
    target_include_directories(${target_name} PUBLIC ${INKER_GGML_INCLUDE_DIR})
    target_link_directories(${target_name} PUBLIC ${INKER_GGML_LIB_DIR})
    target_link_libraries(${target_name} PUBLIC ggml ggml-base ggml-cpu Threads::Threads)
    
    if(GGML_HAS_METAL)
        target_link_libraries(${target_name} PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
    endif()
    
    if(GGML_HAS_CUDA)
        # 透传 CUDA 宏，让 cpp 源码能启用 ggml_backend_cuda_init
        target_compile_definitions(${target_name} PUBLIC GGML_USE_CUDA)
        target_link_libraries(${target_name} PUBLIC 
            ggml-cuda 
            CUDA::cudart 
            CUDA::cublas 
            CUDA::cublasLt
        )
    endif()
endmacro()
# ========================================================================


# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
inker_link_ggml_backend(text_tokenizer)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
inker_link_ggml_backend(tts_transformer)

if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
inker_link_ggml_backend(audio_tokenizer_encoder)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
inker_link_ggml_backend(audio_tokenizer_decoder)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
)
if(WIN32)
    target_link_libraries(qwen3_tts PUBLIC psapi)
endif()

# Shared library with C API (for Nim FFI)
add_library(qwen3tts_shared SHARED
    src/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
)

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Install targets
install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
