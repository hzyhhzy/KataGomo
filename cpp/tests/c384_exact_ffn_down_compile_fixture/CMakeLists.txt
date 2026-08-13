cmake_minimum_required(VERSION 3.18)
project(c384_exact_ffn_down_contract LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(c384_exact_ffn_down_contract
  "${CMAKE_CURRENT_LIST_DIR}/../c384_exact_ffn_down_contract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cuda_specialized/sm120/c384/fixed_batch/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down_registry.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down_registry_stub.cu")
target_compile_features(c384_exact_ffn_down_contract PRIVATE cxx_std_17)
set_target_properties(c384_exact_ffn_down_contract PROPERTIES
  CUDA_ARCHITECTURES 120
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)
target_include_directories(c384_exact_ffn_down_contract PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../..")
target_link_libraries(c384_exact_ffn_down_contract PRIVATE CUDA::cudart)

add_executable(c384_exact_ffn_down_generated_provider_contract
  "${CMAKE_CURRENT_LIST_DIR}/../c384_exact_ffn_down_generated_provider_contract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../c384_exact_ffn_down_generated_provider_fixture.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down_registry.cu")
target_compile_features(c384_exact_ffn_down_generated_provider_contract PRIVATE
  cxx_std_17)
set_target_properties(c384_exact_ffn_down_generated_provider_contract PROPERTIES
  CUDA_ARCHITECTURES 120
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)
target_include_directories(c384_exact_ffn_down_generated_provider_contract PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet")
target_link_libraries(c384_exact_ffn_down_generated_provider_contract PRIVATE
  CUDA::cudart)

# Optional compile-only coverage for the actual production integration TU.
# It needs only a cuDNN include directory; no generated winner object is linked.
set(KATAGO_CUDNN_INCLUDE_DIR "" CACHE PATH
  "cuDNN include directory for the cudabackend compile-only contract")
if(NOT KATAGO_CUDNN_INCLUDE_DIR STREQUAL "")
  if(NOT EXISTS "${KATAGO_CUDNN_INCLUDE_DIR}/cudnn.h")
    message(FATAL_ERROR "KATAGO_CUDNN_INCLUDE_DIR does not contain cudnn.h")
  endif()
  add_library(c384_cudabackend_compile_contract OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cudabackend.cpp")
  set_target_properties(c384_cudabackend_compile_contract PROPERTIES
    EXCLUDE_FROM_ALL TRUE)
  target_compile_features(c384_cudabackend_compile_contract PRIVATE cxx_std_17)
  target_compile_definitions(c384_cudabackend_compile_contract PRIVATE
    USE_CUDA_BACKEND
    CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION}
    KATAGO_ENABLE_C384_EXACT_FIXED_AOT=1
    KATAGO_ENABLE_RENJU15_RMS_SM120=1
    KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120=1
    KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120=1
    KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120=1
    KATAGO_ENABLE_RENJU15_FA4_SM120=1)
  target_include_directories(c384_cudabackend_compile_contract SYSTEM PRIVATE
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${KATAGO_CUDNN_INCLUDE_DIR}")
  target_include_directories(c384_cudabackend_compile_contract PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${CMAKE_CURRENT_LIST_DIR}/../../external")

  # The shared dual-FFN family is independently configurable. Keep a compile
  # contract for DUAL=1/EXACT=0 so exact-transaction bookkeeping cannot leak
  # across its feature guard.
  add_library(c384_cudabackend_dual_only_compile_contract OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/../../neuralnet/cudabackend.cpp")
  set_target_properties(c384_cudabackend_dual_only_compile_contract PROPERTIES
    EXCLUDE_FROM_ALL TRUE)
  target_compile_features(c384_cudabackend_dual_only_compile_contract PRIVATE
    cxx_std_17)
  target_compile_definitions(c384_cudabackend_dual_only_compile_contract PRIVATE
    USE_CUDA_BACKEND
    CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION}
    KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120=1)
  target_include_directories(c384_cudabackend_dual_only_compile_contract SYSTEM PRIVATE
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${KATAGO_CUDNN_INCLUDE_DIR}")
  target_include_directories(c384_cudabackend_dual_only_compile_contract PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${CMAKE_CURRENT_LIST_DIR}/../../external")
endif()
