cmake_minimum_required(VERSION 3.26)

project(
  mosaicvram
  VERSION 0.1.0
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

option(MOSAICVRAM_ENABLE_ONNX "Enable ONNX Runtime backend" OFF)
option(MOSAICVRAM_ENABLE_LLAMA "Enable llama.cpp warm-state command" OFF)
option(MOSAICVRAM_ENABLE_TOKENIZER_JSON
       "Enable native Hugging Face tokenizer.json support" OFF)
set(MOSAICVRAM_ONNXRUNTIME_DIR
    ""
    CACHE PATH "Path to extracted ONNX Runtime package")
set(MOSAICVRAM_LLAMA_DIR
    ""
    CACHE PATH "Path to llama.cpp source directory")
set(MOSAICVRAM_LLAMA_BUILD_DIR
    ""
    CACHE PATH "Path to llama.cpp build directory")
set(MOSAICVRAM_TOKENIZERS_CPP_DIR
    ""
    CACHE PATH "Path to tokenizers-cpp source directory")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_executable(
  mosaicvram
  src/alloc/gpu_allocator.cc
  src/backend/backend_runtime.cc
  src/commands/alloc_smoke.cc
  src/commands/residency_run.cc
  src/commands/run_command.cc
  src/commands/shell_command.cc
  src/cuda/cuda_device.cc
  src/cuda/cuda_error.cc
  src/cuda/cuda_stream.cc
  src/main.cc
  src/residency/residency_controller.cc
  src/runtime/config_line.cc
  src/util/timer.cc)

if(MOSAICVRAM_ENABLE_ONNX)
  if(NOT MOSAICVRAM_ONNXRUNTIME_DIR)
    message(FATAL_ERROR "MOSAICVRAM_ONNXRUNTIME_DIR is required when ONNX is enabled")
  endif()

  target_sources(mosaicvram PRIVATE src/backend/onnx_backend.cc
                                    src/residency/onnx_llm_residency_adapter.cc)
  target_compile_definitions(mosaicvram PRIVATE MOSAICVRAM_ENABLE_ONNX)
  target_include_directories(mosaicvram
                             PRIVATE "${MOSAICVRAM_ONNXRUNTIME_DIR}/include")
  target_link_directories(mosaicvram PRIVATE "${MOSAICVRAM_ONNXRUNTIME_DIR}/lib")
  target_link_libraries(mosaicvram PRIVATE onnxruntime)
endif()

if(MOSAICVRAM_ENABLE_TOKENIZER_JSON)
  if(NOT MOSAICVRAM_ENABLE_ONNX)
    message(FATAL_ERROR "MOSAICVRAM_ENABLE_ONNX is required for tokenizer.json support")
  endif()
  if(NOT MOSAICVRAM_TOKENIZERS_CPP_DIR)
    message(FATAL_ERROR "MOSAICVRAM_TOKENIZERS_CPP_DIR is required when tokenizer.json support is enabled")
  endif()

  set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER
      OFF
      CACHE BOOL "Disable SentencePiece for MosaicVRAM tokenizer.json phase")
  if(MSVC)
    set(TOKENIZERS_CPP_MSVC_RUNTIME_LIBRARY
        "MD"
        CACHE STRING "Match MosaicVRAM MSVC runtime")
  endif()
  add_subdirectory("${MOSAICVRAM_TOKENIZERS_CPP_DIR}" tokenizers-cpp
                   EXCLUDE_FROM_ALL)
  target_sources(mosaicvram PRIVATE src/tokenizer/tokenizers_cpp_adapter.cc)
  target_compile_definitions(mosaicvram
                             PRIVATE MOSAICVRAM_ENABLE_TOKENIZER_JSON)
  target_link_libraries(mosaicvram PRIVATE tokenizers_cpp)
endif()

if(MOSAICVRAM_ENABLE_LLAMA)
  if(NOT MOSAICVRAM_LLAMA_DIR)
    message(FATAL_ERROR "MOSAICVRAM_LLAMA_DIR is required when llama.cpp is enabled")
  endif()
  if(NOT MOSAICVRAM_LLAMA_BUILD_DIR)
    message(FATAL_ERROR "MOSAICVRAM_LLAMA_BUILD_DIR is required when llama.cpp is enabled")
  endif()

  target_sources(mosaicvram PRIVATE src/commands/llama_state_smoke.cc
                                    src/residency/llama_residency_adapter.cc)
  target_compile_definitions(mosaicvram PRIVATE MOSAICVRAM_ENABLE_LLAMA)
  target_include_directories(
    mosaicvram PRIVATE "${MOSAICVRAM_LLAMA_DIR}/include"
                        "${MOSAICVRAM_LLAMA_DIR}/ggml/include")
  target_link_directories(mosaicvram PRIVATE "${MOSAICVRAM_LLAMA_BUILD_DIR}/src")
  target_link_libraries(mosaicvram PRIVATE llama)
endif()

target_compile_features(mosaicvram PRIVATE cxx_std_20)
target_include_directories(mosaicvram PRIVATE src)
target_link_libraries(mosaicvram PRIVATE CUDA::cudart)

if(MSVC)
  target_compile_options(mosaicvram PRIVATE /W4 /permissive-)
else()
  target_compile_options(mosaicvram PRIVATE -Wall -Wextra -Wpedantic)
endif()
