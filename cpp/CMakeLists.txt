cmake_minimum_required(VERSION 3.18.2)
project(katago)

set(CMAKE_CXX_STANDARD 17)

set(FOUR_PROFILE_PROVIDER_SOURCES
  neuralnet/four_profile/provider_v1.cpp
  neuralnet/four_profile/mode.cpp
  neuralnet/four_profile/registry.cpp
  neuralnet/four_profile/manager.cpp
  neuralnet/four_profile/native_model_view.cpp
  neuralnet/four_profile/p1_factory.cpp
  neuralnet/four_profile/p2_factory.cpp
  neuralnet/four_profile/stub_factories.cpp)

add_executable(four_profile_provider_contract EXCLUDE_FROM_ALL
  tests/four_profile_provider_contract.cpp
  ${FOUR_PROFILE_PROVIDER_SOURCES})
target_include_directories(four_profile_provider_contract PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(four_profile_provider_contract_check
  COMMAND $<TARGET_FILE:four_profile_provider_contract>
  DEPENDS four_profile_provider_contract
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM)

add_executable(forbidden_bulk_map_contract EXCLUDE_FROM_ALL
  tests/forbidden_bulk_map_contract.cpp
  forbiddenPoint/ForbiddenPointFinder.cpp
  forbiddenPoint/ForbiddenPointFinderBulk.cpp)
target_include_directories(forbidden_bulk_map_contract PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(forbidden_bulk_map_contract PRIVATE
  KATAGO_FORBIDDEN_BULK_TEST_HELPERS=1)
if(USE_BIGGER_BOARDS_EXPENSIVE)
  target_compile_definitions(forbidden_bulk_map_contract PRIVATE COMPILE_MAX_BOARD_LEN=29)
endif()

include_directories(external)
include_directories(external/tclap-1.2.2/include)
include_directories(SYSTEM external/filesystem-1.5.8/include) #SYSTEM suppresses a few warnings

#--------------------------- PLATFORM SPECIFIC -------------------------------------------------------------------------

if(APPLE)
  # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset   "${Esc}[m")
  set(ColorBold    "${Esc}[1m")
  set(ColorRed     "${Esc}[31m")
  set(ColorBoldRed "${ColorRed}${ColorBold}")
endif()

#--------------------------- CMAKE VARIABLES (partly for Cmake GUI) ----------------------------------------------------

set(BUILD_DISTRIBUTED 0 CACHE BOOL "Build with http support for contributing to distributed training")
set(USE_BACKEND CACHE STRING "Neural net backend")
string(TOUPPER "${USE_BACKEND}" USE_BACKEND)
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA TENSORRT OPENCL EIGEN ONNX-CPU ONNX-DIRECTML)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(USE_AVX2 0 CACHE BOOL "Compile with AVX2")
set(USE_BIGGER_BOARDS_EXPENSIVE 0 CACHE BOOL "Allow boards up to size 29. Compiling with this will use more memory and slow down KataGo, even when playing on boards of size 19.")
set(KATAGO_BUILD_BENCHMARKNN 0 CACHE BOOL "Build the test-only raw neural-net benchmark command")
set(KATAGO_BUILD_V105_WIRE_CONTRACT 0 CACHE BOOL "Build the Release-safe native v105 wire contract")
set(KATAGO_BUILD_NNRAWGATE 0 CACHE BOOL "Build the test-only cross-implementation raw neural-net gate command")
set(KATAGO_ENABLE_P1_SM120 0 CACHE BOOL
  "Use the qualified C256/H8 B36 SM120 provider when its exact dependencies are present")
set(KATAGO_P1_SM120_AVAILABLE 0)
set(KATAGO_ENABLE_P2_SM120 0 CACHE BOOL
  "Use the C256/H8 B28/S361 SM120 provider with an external FA4 package")
set(KATAGO_P2_SM120_AVAILABLE 0)
set(KATAGO_P2_FA4_PACKAGE_CMAKE "" CACHE FILEPATH
  "CMake manifest for the C256/H8 B28/S361 FA4 AOT package")
set(KATAGO_ENABLE_P3_PROVIDER 0 CACHE BOOL
  "Build the default-off SM120 C384/H12 B28 v105 FP16 provider")
set(KATAGO_ENABLE_P4_PROVIDER 0 CACHE BOOL
  "Build the SM120 C384/H12 B28 v105 INT8 provider (uses the P3 FA4 package)")
set(KATAGO_CUTLASS_ROOT "" CACHE PATH
  "CUTLASS checkout pinned to dcf215af68a2d08d305076c152a06f201728cd53")
set(KATAGO_P3_EXACT_PACKAGE_CMAKE "" CACHE FILEPATH
  "SHA-locked C384 B28 QKV package manifest for P3")
set(KATAGO_P3_FA4_PACKAGE_CMAKE "" CACHE FILEPATH
  "SHA-locked C384/H12 B28 FA4 package manifest for P3")
set(KATAGO_P3_DOWN_PACKAGE_CMAKE "" CACHE FILEPATH
  "SHA-locked C384 B28 FFN-down package manifest for P3")

#--------------------------- NEURAL NET BACKEND ------------------------------------------------------------------------

message(STATUS "Building 'katago' executable for GTP engine and other tools.")
if(USE_BACKEND STREQUAL "CUDA")
  message(STATUS "-DUSE_BACKEND=CUDA, using CUDA backend.")

  enable_language(CUDA)

  set(CUDA_STANDARD 11)
  if(KATAGO_ENABLE_P1_SM120)
    set(KATAGO_P1_PINNED_CUTLASS_COMMIT
      "dcf215af68a2d08d305076c152a06f201728cd53")
    set(KATAGO_P1_FA4_ROOT
      "${CMAKE_CURRENT_SOURCE_DIR}/external/renju15-sm120-fa4")
    set(KATAGO_P1_FA4_HEADER
      "${KATAGO_P1_FA4_ROOT}/renju15_fa4_b36_tn128.h")
    set(KATAGO_P1_FA4_OBJECT
      "${KATAGO_P1_FA4_ROOT}/renju15_fa4_b36_tn128.o")
    set(KATAGO_P1_FA4_METADATA
      "${KATAGO_P1_FA4_ROOT}/renju15_fa4_b36_tn128.json")
    set(KATAGO_P1_FA4_RUNTIME
      "${KATAGO_P1_FA4_ROOT}/libcuda_dialect_runtime_static.a")
    set(KATAGO_P1_DEPENDENCY_REASON "")

    if(WIN32)
      set(KATAGO_P1_DEPENDENCY_REASON
        "the SHA-locked B36 FA4 package contains Linux ELF objects")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      set(KATAGO_P1_DEPENDENCY_REASON "CUDA 13.0 or newer is required")
    elseif(NOT EXISTS "${KATAGO_CUTLASS_ROOT}/include/cutlass/cutlass.h" OR
           NOT EXISTS "${KATAGO_CUTLASS_ROOT}/examples/45_dual_gemm/device/dual_gemm.h")
      set(KATAGO_P1_DEPENDENCY_REASON
        "KATAGO_CUTLASS_ROOT does not contain the pinned CUTLASS headers")
    elseif(NOT EXISTS "${KATAGO_P1_FA4_HEADER}" OR
           NOT EXISTS "${KATAGO_P1_FA4_OBJECT}" OR
           NOT EXISTS "${KATAGO_P1_FA4_METADATA}" OR
           NOT EXISTS "${KATAGO_P1_FA4_RUNTIME}")
      set(KATAGO_P1_DEPENDENCY_REASON "the SHA-locked B36 FA4 package is incomplete")
    else()
      find_package(Git QUIET)
      if(NOT GIT_FOUND)
        set(KATAGO_P1_DEPENDENCY_REASON "Git is required to verify the CUTLASS revision")
      else()
        execute_process(
          COMMAND "${GIT_EXECUTABLE}" -C "${KATAGO_CUTLASS_ROOT}" rev-parse HEAD
          RESULT_VARIABLE KATAGO_P1_CUTLASS_REVISION_RESULT
          OUTPUT_VARIABLE KATAGO_P1_CUTLASS_REVISION
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT KATAGO_P1_CUTLASS_REVISION_RESULT EQUAL 0 OR
           NOT KATAGO_P1_CUTLASS_REVISION STREQUAL KATAGO_P1_PINNED_CUTLASS_COMMIT)
          set(KATAGO_P1_DEPENDENCY_REASON
            "CUTLASS revision is '${KATAGO_P1_CUTLASS_REVISION}', expected '${KATAGO_P1_PINNED_CUTLASS_COMMIT}'")
        else()
          # HEAD alone is not an exact dependency lock: tracked edits under
          # either include root can change every compiled P1 CUTLASS kernel
          # while rev-parse still reports the pinned commit. Unrelated docs
          # and examples stay outside the qualification boundary.
          execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${KATAGO_CUTLASS_ROOT}"
              diff --quiet HEAD -- include examples/45_dual_gemm
            RESULT_VARIABLE KATAGO_P1_CUTLASS_DIFF_RESULT)
          if(NOT KATAGO_P1_CUTLASS_DIFF_RESULT EQUAL 0)
            set(KATAGO_P1_DEPENDENCY_REASON
              "tracked CUTLASS inputs under include or examples/45_dual_gemm differ from the pinned commit")
          endif()
        endif()
      endif()
    endif()

    if(KATAGO_P1_DEPENDENCY_REASON STREQUAL "")
      file(SHA256 "${KATAGO_P1_FA4_HEADER}" KATAGO_P1_FA4_HEADER_SHA)
      file(SHA256 "${KATAGO_P1_FA4_OBJECT}" KATAGO_P1_FA4_OBJECT_SHA)
      file(SHA256 "${KATAGO_P1_FA4_METADATA}" KATAGO_P1_FA4_METADATA_SHA)
      file(SHA256 "${KATAGO_P1_FA4_RUNTIME}" KATAGO_P1_FA4_RUNTIME_SHA)
      if(NOT KATAGO_P1_FA4_HEADER_SHA STREQUAL
           "48e8572e7dda80f7022085e0384830e88f3a4f58e6a79f25c40b0ff60167311d" OR
         NOT KATAGO_P1_FA4_OBJECT_SHA STREQUAL
           "690fbd68a750e90ad4ea964ecb729258ccce8d5d81a8e996dc436435dcb62a9d" OR
         NOT KATAGO_P1_FA4_METADATA_SHA STREQUAL
           "08c8b0121d15c75c47dcc6dde987eff4dba64840989a301091063c57e518e925" OR
         NOT KATAGO_P1_FA4_RUNTIME_SHA STREQUAL
           "5ffeb7ae24bf02755a6ad751d196b2f1efd18a834b747972ffdd8bbc6550b899")
        set(KATAGO_P1_DEPENDENCY_REASON "the B36 FA4 package SHA256 contract failed")
      endif()
    endif()

    if(KATAGO_P1_DEPENDENCY_REASON STREQUAL "")
      set(KATAGO_P1_SM120_AVAILABLE 1)
      message(STATUS "P1 SM120 provider available: C256/H8/B36/S225, dynamic N/N")
    else()
      message(STATUS
        "P1 SM120 provider unavailable; exact matches use official CUDA: ${KATAGO_P1_DEPENDENCY_REASON}")
    endif()
  else()
    message(STATUS "P1 SM120 provider disabled; exact matches use official CUDA")
  endif()

  if(KATAGO_ENABLE_P2_SM120)
    if(NOT KATAGO_P1_SM120_AVAILABLE)
      message(STATUS "P2 provider unavailable: enable the shared P1 C256 SM120 kernel build")
    elseif(KATAGO_P2_FA4_PACKAGE_CMAKE STREQUAL "" OR
           NOT EXISTS "${KATAGO_P2_FA4_PACKAGE_CMAKE}")
      message(STATUS "P2 provider unavailable: KATAGO_P2_FA4_PACKAGE_CMAKE is missing")
    else()
      include("${KATAGO_P2_FA4_PACKAGE_CMAKE}")
      if(NOT KATAGO_C256_S361_FA4_SCHEMA STREQUAL "1" OR
         NOT KATAGO_C256_S361_FA4_BATCH STREQUAL "28" OR
         NOT KATAGO_C256_S361_FA4_SEQUENCE STREQUAL "361" OR
         NOT KATAGO_C256_S361_FA4_HEADS STREQUAL "8")
        message(FATAL_ERROR "P2 FA4 package does not describe B28/S361/H8")
      endif()
      foreach(KATAGO_P2_FILE IN ITEMS
          "${KATAGO_C256_S361_FA4_HEADER}"
          "${KATAGO_C256_S361_FA4_OBJECT}"
          "${KATAGO_C256_S361_FA4_BRIDGE}"
          "${KATAGO_C256_S361_FA4_RUNTIME}")
        if(NOT EXISTS "${KATAGO_P2_FILE}")
          message(FATAL_ERROR "P2 FA4 package file is missing: ${KATAGO_P2_FILE}")
        endif()
      endforeach()
      get_filename_component(KATAGO_P2_FA4_INCLUDE_DIR
        "${KATAGO_C256_S361_FA4_HEADER}" DIRECTORY)
      set(KATAGO_P2_SM120_AVAILABLE 1)
      message(STATUS "P2 SM120 provider available: C256/H8/B28/S361, dynamic N/N")
    endif()
  endif()

  set(NEURALNET_BACKEND_SOURCES
    neuralnet/cudabackend.cpp
    neuralnet/cudautils.cpp
    neuralnet/cudahelpers.cu
    neuralnet/debugprint.cpp
  )
  if(KATAGO_P1_SM120_AVAILABLE)
    add_library(katago_p1_sm120_kernels OBJECT
      neuralnet/cuda_specialized/sm120/shared/rms_norm.cu
      neuralnet/cuda_specialized/sm120/shared/residual_gemm.cu
      neuralnet/cuda_specialized/sm120/c256/fixed_batch/qkv_rope_gemm.cu
      neuralnet/cudafusedffn.cu)
    set_target_properties(katago_p1_sm120_kernels PROPERTIES
      CUDA_ARCHITECTURES "120"
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON)
    target_compile_options(katago_p1_sm120_kernels PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    target_compile_definitions(katago_p1_sm120_kernels PRIVATE
      KATAGO_ENABLE_RENJU15_RMS_SM120=1
      KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120=1)
    target_include_directories(katago_p1_sm120_kernels PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet")
    target_include_directories(katago_p1_sm120_kernels SYSTEM PRIVATE
      "${KATAGO_CUTLASS_ROOT}/include"
      "${KATAGO_CUTLASS_ROOT}/examples/45_dual_gemm")
    if(KATAGO_P2_SM120_AVAILABLE)
      target_sources(katago_p1_sm120_kernels PRIVATE
        "${KATAGO_C256_S361_FA4_BRIDGE}")
      target_include_directories(katago_p1_sm120_kernels PRIVATE
        "${KATAGO_P2_FA4_INCLUDE_DIR}")
    endif()
    list(APPEND NEURALNET_BACKEND_SOURCES
      neuralnet/four_profile/p1_provider.cpp
      neuralnet/cuda_specialized/sm120/c256/fixed_batch/fa4.cpp
      $<TARGET_OBJECTS:katago_p1_sm120_kernels>)
  endif()
  # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 88 89 90 100 110 103 120 121)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
      set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 75 80 86 87 90 120)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
      set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 90)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
      set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87 90)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5)
      set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
      set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
      set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2)
      set(CMAKE_CUDA_ARCHITECTURES 30 35 37 50 52 53 60 61 62 70 72 75)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    else()
      message(WARNING "CUDA 10.2 or greater is recommended, but attempting to build anyways")
      set(CMAKE_CUDA_ARCHITECTURES 30 37 53 70)
    endif()
  endif()

  set(KATAGO_P3_PROVIDER_COMPILED 0)
  set(KATAGO_P4_PROVIDER_COMPILED 0)
  set(KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE 0)
  set(KATAGO_P3_EXTERNAL_LINK_LIBRARIES "")
  if(KATAGO_ENABLE_P3_PROVIDER)
    set(KATAGO_P3_PINNED_CUTLASS_COMMIT
      "dcf215af68a2d08d305076c152a06f201728cd53")
    set(KATAGO_P3_CUTLASS_READY 0)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(STATUS "P3 provider unavailable: CUDA 13.0 or newer is required")
    elseif(NOT EXISTS "${KATAGO_CUTLASS_ROOT}/include/cutlass/cutlass.h" OR
           NOT EXISTS "${KATAGO_CUTLASS_ROOT}/examples/45_dual_gemm/device/dual_gemm.h")
      message(STATUS "P3 provider unavailable: pinned CUTLASS checkout is missing")
    else()
      find_package(Git QUIET)
      if(GIT_FOUND)
        execute_process(
          COMMAND "${GIT_EXECUTABLE}" -C "${KATAGO_CUTLASS_ROOT}" rev-parse HEAD
          RESULT_VARIABLE KATAGO_P3_CUTLASS_REVISION_RESULT
          OUTPUT_VARIABLE KATAGO_P3_CUTLASS_REVISION
          OUTPUT_STRIP_TRAILING_WHITESPACE)
      endif()
      if(NOT GIT_FOUND OR NOT KATAGO_P3_CUTLASS_REVISION_RESULT EQUAL 0 OR
         NOT KATAGO_P3_CUTLASS_REVISION STREQUAL KATAGO_P3_PINNED_CUTLASS_COMMIT)
        message(STATUS
          "P3 provider unavailable: CUTLASS must be pinned to ${KATAGO_P3_PINNED_CUTLASS_COMMIT}")
      else()
        set(KATAGO_P3_CUTLASS_READY 1)
      endif()
    endif()

    if(KATAGO_P3_CUTLASS_READY)
      set(KATAGO_P3_PROVIDER_COMPILED 1)
      set(KATAGO_P3_QKV_TACTIC_ID "")
      set(KATAGO_P3_FFN_DOWN_TACTIC_ID "")
      set(KATAGO_P3_EXACT_REGISTRY
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/registry_stub.cu)
      set(KATAGO_P3_FA4_REGISTRY
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/fa4_registry_stub.cpp)
      set(KATAGO_P3_DOWN_REGISTRY
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down_registry_stub.cu)
      set(KATAGO_P3_PACKAGE_BRIDGES "")
      set(KATAGO_P3_PACKAGE_INCLUDE_DIRS "")

      set(KATAGO_P3_PACKAGE_COUNT 0)
      foreach(KATAGO_P3_MANIFEST IN ITEMS
          "${KATAGO_P3_EXACT_PACKAGE_CMAKE}"
          "${KATAGO_P3_FA4_PACKAGE_CMAKE}"
          "${KATAGO_P3_DOWN_PACKAGE_CMAKE}")
        if(NOT KATAGO_P3_MANIFEST STREQUAL "")
          math(EXPR KATAGO_P3_PACKAGE_COUNT "${KATAGO_P3_PACKAGE_COUNT} + 1")
        endif()
      endforeach()

      if(KATAGO_P3_PACKAGE_COUNT EQUAL 3)
        foreach(KATAGO_P3_MANIFEST_VAR IN ITEMS
            KATAGO_P3_EXACT_PACKAGE_CMAKE
            KATAGO_P3_FA4_PACKAGE_CMAKE
            KATAGO_P3_DOWN_PACKAGE_CMAKE)
          get_filename_component(${KATAGO_P3_MANIFEST_VAR}
            "${${KATAGO_P3_MANIFEST_VAR}}" ABSOLUTE)
          if(NOT EXISTS "${${KATAGO_P3_MANIFEST_VAR}}")
            message(FATAL_ERROR "P3 package manifest is missing: ${${KATAGO_P3_MANIFEST_VAR}}")
          endif()
        endforeach()
        file(SHA256 "${KATAGO_P3_EXACT_PACKAGE_CMAKE}" KATAGO_P3_EXACT_MANIFEST_SHA)
        file(SHA256 "${KATAGO_P3_FA4_PACKAGE_CMAKE}" KATAGO_P3_FA4_MANIFEST_SHA)
        file(SHA256 "${KATAGO_P3_DOWN_PACKAGE_CMAKE}" KATAGO_P3_DOWN_MANIFEST_SHA)
        if(NOT KATAGO_P3_EXACT_MANIFEST_SHA STREQUAL
             "cadf3c674d27c3f2a4d91be535ff75bb71396d474c56d526046f811bfec422ae" OR
           NOT KATAGO_P3_FA4_MANIFEST_SHA STREQUAL
             "a277e078d44611dc946e101bfe153d9701aa64a0f2ffca48a6889f3f870becf0" OR
           NOT KATAGO_P3_DOWN_MANIFEST_SHA STREQUAL
             "f3d05ccb50e4e57c70a8504ad91bd0359999cc17e4800d924a7e3cacc75e86fe")
          message(FATAL_ERROR "P3 package manifest SHA256 mismatch")
        endif()

        include("${KATAGO_P3_EXACT_PACKAGE_CMAKE}")
        include("${KATAGO_P3_FA4_PACKAGE_CMAKE}")
        include("${KATAGO_P3_DOWN_PACKAGE_CMAKE}")
        if(NOT KATAGO_C384_EXACT_AOT_PACKAGE_SCHEMA STREQUAL "1" OR
           NOT KATAGO_C384_EXACT_AOT_PACKAGE_MODE STREQUAL "PRODUCTION" OR
           NOT KATAGO_C384_EXACT_AOT_SELECTED_BATCH STREQUAL "28" OR
           NOT KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID STREQUAL
             "c384-s225-h12d32-qkv1152-rope192-m128n128k32-a4x2-sauto-g170-b28-abi1" OR
           NOT KATAGO_C384_H12_FA4_PACKAGE_SCHEMA STREQUAL "1" OR
           NOT KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "PRODUCTION" OR
           NOT KATAGO_C384_H12_FA4_BATCHES STREQUAL "28" OR
           NOT KATAGO_C384_FFN_DOWN_PACKAGE_SCHEMA STREQUAL "1" OR
           NOT KATAGO_C384_FFN_DOWN_PACKAGE_MODE STREQUAL "PRODUCTION" OR
           NOT KATAGO_C384_FFN_DOWN_SELECTED_BATCH STREQUAL "28" OR
           NOT KATAGO_C384_FFN_DOWN_SELECTED_ID STREQUAL
             "c384-s225-residual-ffn-down-m6300-k1024-n384-m128n128k32s3-natural-b28-abi1")
          message(FATAL_ERROR "P3 package shape or selected tactic does not match B28 P3")
        endif()

        set(KATAGO_P3_HASH_PAIRS
          ${KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256}
          ${KATAGO_C384_FFN_DOWN_GENERATED_FILE_SHA256})
        list(LENGTH KATAGO_P3_HASH_PAIRS KATAGO_P3_HASH_PAIR_LENGTH)
        math(EXPR KATAGO_P3_HASH_PAIR_REMAINDER "${KATAGO_P3_HASH_PAIR_LENGTH} % 2")
        if(KATAGO_P3_HASH_PAIR_LENGTH EQUAL 0 OR NOT KATAGO_P3_HASH_PAIR_REMAINDER EQUAL 0)
          message(FATAL_ERROR "P3 exact/down package SHA list is malformed")
        endif()
        math(EXPR KATAGO_P3_HASH_PAIR_LAST "${KATAGO_P3_HASH_PAIR_LENGTH} - 2")
        foreach(KATAGO_P3_HASH_INDEX RANGE 0 ${KATAGO_P3_HASH_PAIR_LAST} 2)
          math(EXPR KATAGO_P3_HASH_VALUE_INDEX "${KATAGO_P3_HASH_INDEX} + 1")
          list(GET KATAGO_P3_HASH_PAIRS ${KATAGO_P3_HASH_INDEX} KATAGO_P3_HASH_FILE)
          list(GET KATAGO_P3_HASH_PAIRS ${KATAGO_P3_HASH_VALUE_INDEX} KATAGO_P3_HASH_EXPECTED)
          if(NOT EXISTS "${KATAGO_P3_HASH_FILE}")
            message(FATAL_ERROR "P3 package artifact is missing: ${KATAGO_P3_HASH_FILE}")
          endif()
          file(SHA256 "${KATAGO_P3_HASH_FILE}" KATAGO_P3_HASH_ACTUAL)
          if(NOT KATAGO_P3_HASH_ACTUAL STREQUAL KATAGO_P3_HASH_EXPECTED)
            message(FATAL_ERROR "P3 package artifact SHA256 mismatch: ${KATAGO_P3_HASH_FILE}")
          endif()
        endforeach()
        list(LENGTH KATAGO_C384_H12_FA4_ASSETS KATAGO_P3_FA4_ASSET_COUNT)
        list(LENGTH KATAGO_C384_H12_FA4_ASSET_SHAS KATAGO_P3_FA4_SHA_COUNT)
        if(NOT KATAGO_P3_FA4_ASSET_COUNT EQUAL KATAGO_P3_FA4_SHA_COUNT OR
           KATAGO_P3_FA4_ASSET_COUNT EQUAL 0)
          message(FATAL_ERROR "P3 FA4 package SHA list is malformed")
        endif()
        math(EXPR KATAGO_P3_FA4_ASSET_LAST "${KATAGO_P3_FA4_ASSET_COUNT} - 1")
        foreach(KATAGO_P3_FA4_INDEX RANGE 0 ${KATAGO_P3_FA4_ASSET_LAST})
          list(GET KATAGO_C384_H12_FA4_ASSETS ${KATAGO_P3_FA4_INDEX} KATAGO_P3_HASH_FILE)
          list(GET KATAGO_C384_H12_FA4_ASSET_SHAS ${KATAGO_P3_FA4_INDEX} KATAGO_P3_HASH_EXPECTED)
          if(NOT EXISTS "${KATAGO_P3_HASH_FILE}")
            message(FATAL_ERROR "P3 FA4 artifact is missing: ${KATAGO_P3_HASH_FILE}")
          endif()
          file(SHA256 "${KATAGO_P3_HASH_FILE}" KATAGO_P3_HASH_ACTUAL)
          if(NOT KATAGO_P3_HASH_ACTUAL STREQUAL KATAGO_P3_HASH_EXPECTED)
            message(FATAL_ERROR "P3 FA4 artifact SHA256 mismatch: ${KATAGO_P3_HASH_FILE}")
          endif()
        endforeach()
        foreach(KATAGO_P3_FA4_SINGLE IN ITEMS REGISTRY RUNTIME)
          set(KATAGO_P3_HASH_FILE "${KATAGO_C384_H12_FA4_${KATAGO_P3_FA4_SINGLE}}")
          set(KATAGO_P3_HASH_EXPECTED "${KATAGO_C384_H12_FA4_${KATAGO_P3_FA4_SINGLE}_SHA}")
          if(NOT EXISTS "${KATAGO_P3_HASH_FILE}")
            message(FATAL_ERROR "P3 FA4 ${KATAGO_P3_FA4_SINGLE} is missing")
          endif()
          file(SHA256 "${KATAGO_P3_HASH_FILE}" KATAGO_P3_HASH_ACTUAL)
          if(NOT KATAGO_P3_HASH_ACTUAL STREQUAL KATAGO_P3_HASH_EXPECTED)
            message(FATAL_ERROR "P3 FA4 ${KATAGO_P3_FA4_SINGLE} SHA256 mismatch")
          endif()
        endforeach()

        set(KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE 1)
        set(KATAGO_P3_QKV_TACTIC_ID "${KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID}")
        set(KATAGO_P3_FFN_DOWN_TACTIC_ID "${KATAGO_C384_FFN_DOWN_SELECTED_ID}")
        set(KATAGO_P3_EXACT_REGISTRY "${KATAGO_C384_EXACT_AOT_REGISTRY_PROVIDER}")
        set(KATAGO_P3_FA4_REGISTRY "${KATAGO_C384_H12_FA4_REGISTRY}")
        set(KATAGO_P3_DOWN_REGISTRY "${KATAGO_C384_FFN_DOWN_REGISTRY_PROVIDER}")
        set(KATAGO_P3_PACKAGE_BRIDGES
          ${KATAGO_C384_EXACT_AOT_GENERATED_BRIDGES}
          ${KATAGO_C384_H12_FA4_BRIDGES}
          ${KATAGO_C384_FFN_DOWN_GENERATED_BRIDGES})
        set(KATAGO_P3_PACKAGE_INCLUDE_DIRS
          "${KATAGO_C384_EXACT_AOT_GENERATED_INCLUDE_DIR}"
          "${KATAGO_C384_FFN_DOWN_GENERATED_INCLUDE_DIR}")
        foreach(KATAGO_P3_FA4_HEADER IN LISTS KATAGO_C384_H12_FA4_HEADERS)
          get_filename_component(KATAGO_P3_FA4_HEADER_DIR
            "${KATAGO_P3_FA4_HEADER}" DIRECTORY)
          list(APPEND KATAGO_P3_PACKAGE_INCLUDE_DIRS "${KATAGO_P3_FA4_HEADER_DIR}")
        endforeach()
        list(REMOVE_DUPLICATES KATAGO_P3_PACKAGE_INCLUDE_DIRS)
        set(KATAGO_P3_EXTERNAL_LINK_LIBRARIES
          ${KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS}
          ${KATAGO_C384_H12_FA4_OBJECTS}
          ${KATAGO_C384_FFN_DOWN_GENERATED_OBJECTS}
          "${KATAGO_C384_H12_FA4_RUNTIME}")
        message(STATUS "P3 provider: linked SHA-locked B28 QKV/FA4/down packages")
      elseif(KATAGO_P3_PACKAGE_COUNT GREATER 0)
        message(STATUS "P3 provider unavailable: all three package manifests are required together")
      else()
        message(STATUS "P3 provider compiled with unavailable package stubs")
      endif()

      set(KATAGO_P3_PROVIDER_SOURCES
        neuralnet/four_profile/p3_provider.cpp
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/plan.cpp
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/weights.cpp
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/registry.cu
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/fa4.cpp
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down.cpp
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down_registry.cu
        neuralnet/cuda_specialized/sm120/c384/fixed_batch/qknorm_rope.cu
        ${KATAGO_P3_EXACT_REGISTRY}
        ${KATAGO_P3_FA4_REGISTRY}
        ${KATAGO_P3_DOWN_REGISTRY}
        ${KATAGO_P3_PACKAGE_BRIDGES})
      if(NOT KATAGO_P1_SM120_AVAILABLE)
        list(APPEND KATAGO_P3_PROVIDER_SOURCES
          neuralnet/cudafusedffn.cu
          neuralnet/cuda_specialized/sm120/shared/rms_norm.cu)
      endif()
      add_library(katago_p3_provider OBJECT ${KATAGO_P3_PROVIDER_SOURCES})
      set_target_properties(katago_p3_provider PROPERTIES
        CUDA_ARCHITECTURES "120"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
      target_compile_options(katago_p3_provider PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
      if(MSVC)
        target_compile_options(katago_p3_provider PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
      endif()
      target_compile_definitions(katago_p3_provider PRIVATE
        KATAGO_ENABLE_RENJU15_RMS_SM120=1
        KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE=${KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE}
        "KATAGO_P3_QKV_TACTIC_ID=\"${KATAGO_P3_QKV_TACTIC_ID}\""
        "KATAGO_P3_FFN_DOWN_TACTIC_ID=\"${KATAGO_P3_FFN_DOWN_TACTIC_ID}\"")
      target_include_directories(katago_p3_provider PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet"
        ${KATAGO_P3_PACKAGE_INCLUDE_DIRS})
      target_include_directories(katago_p3_provider SYSTEM PRIVATE
        "${KATAGO_CUTLASS_ROOT}/include"
        "${KATAGO_CUTLASS_ROOT}/examples/45_dual_gemm")
      list(APPEND NEURALNET_BACKEND_SOURCES $<TARGET_OBJECTS:katago_p3_provider>)

      if(KATAGO_ENABLE_P4_PROVIDER)
        add_library(katago_p4_provider OBJECT
          neuralnet/four_profile/p4_provider.cpp
          neuralnet/cuda_specialized/sm120/c384/experimental_int8/kernels.cu)
        set_target_properties(katago_p4_provider PROPERTIES
          CUDA_ARCHITECTURES "120"
          CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED ON
          CXX_STANDARD 17
          CXX_STANDARD_REQUIRED ON
          POSITION_INDEPENDENT_CODE ON)
        target_compile_options(katago_p4_provider PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
        if(MSVC)
          target_compile_options(katago_p4_provider PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
        endif()
        target_compile_definitions(katago_p4_provider PRIVATE
          KATAGO_P4_EXTERNAL_PACKAGES_AVAILABLE=${KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE})
        target_include_directories(katago_p4_provider PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet"
          ${KATAGO_P3_PACKAGE_INCLUDE_DIRS})
        target_include_directories(katago_p4_provider SYSTEM PRIVATE
          "${KATAGO_CUTLASS_ROOT}/include"
          "${KATAGO_CUTLASS_ROOT}/examples/45_dual_gemm")
        list(APPEND NEURALNET_BACKEND_SOURCES
          $<TARGET_OBJECTS:katago_p4_provider>)
        set(KATAGO_P4_PROVIDER_COMPILED 1)
      endif()
    endif()
  endif()
elseif(USE_BACKEND STREQUAL "TENSORRT")
  message(STATUS "-DUSE_BACKEND=TENSORRT, using TensorRT backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/trtbackend.cpp
    )
elseif(USE_BACKEND STREQUAL "OPENCL")
  message(STATUS "-DUSE_BACKEND=OPENCL, using OpenCL backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/openclbackend.cpp
    neuralnet/openclkernels.cpp
    neuralnet/openclhelpers.cpp
    neuralnet/opencltuner.cpp
    )
elseif(USE_BACKEND STREQUAL "EIGEN")
  message(STATUS "-DUSE_BACKEND=EIGEN, using Eigen CPU backend.")
  if(NOT USE_AVX2)
    message(STATUS "You can also specify USE_AVX2 (-DUSE_AVX2=1 on command line) if you have a modern CPU for better performance.")
  endif()
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/eigenbackend.cpp
    )
elseif(USE_BACKEND STREQUAL "ONNX-CPU")
  message(STATUS "-DUSE_BACKEND=ONNX-CPU, using ONNX Runtime CPU backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/onnxbackend_cpu.cpp
    )
elseif(USE_BACKEND STREQUAL "ONNX-DIRECTML")
  message(STATUS "-DUSE_BACKEND=ONNX-DIRECTML, using ONNX Runtime DirectML backend.")
  if(NOT WIN32)
    message(FATAL_ERROR "ONNX-DIRECTML backend is only supported on Windows.")
  endif()
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/onnxbackend_directml.cpp
    )
elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=TENSORRT or -DUSE_BACKEND=OPENCL or -DUSE_BACKEND=EIGEN to compile with the respective backend.${ColorReset}")
  set(NEURALNET_BACKEND_SOURCES neuralnet/dummybackend.cpp)
else()
  message(FATAL_ERROR "Unrecognized backend: " ${USE_BACKEND})
endif()


#--------------------------- TCMALLOC ----------------------------------------------------------------------------------

if(USE_TCMALLOC)
  message(STATUS "-DUSE_TCMALLOC=1 is set, using tcmalloc as the allocator")
  find_library(TCMALLOC_LIB NAMES tcmalloc_minimal HINTS /usr)
  if(NOT TCMALLOC_LIB)
    message(FATAL_ERROR "Could not find tcmalloc")
  endif()
endif()

# set (Gperftools_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")
# find_package(Gperftools REQUIRED)

#--------------------------- GIT ---------------------------------------------------------------------------------------

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  message(STATUS "-DNO_GIT_REVISION=1 is set, avoiding including the Git revision in compiled executable")
  unset(GIT_HEADER_FILE_ALWAYS_UPDATED)
else()
  if(NO_GIT_REVISION AND BUILD_DISTRIBUTED)
    message(STATUS "${ColorRed}NO_GIT_REVISION is set, but BUILD_DISTRIBUTED is also set and distributed requires git revision, so ignoring NO_GIT_REVISION.${ColorReset}")
  elseif(BUILD_DISTRIBUTED)
    message(STATUS "Including Git revision in the compiled executable")
  else()
    message(STATUS "Including Git revision in the compiled executable, specify -DNO_GIT_REVISION=1 to disable")
  endif()
  find_package(Git)
  if(NOT GIT_FOUND)
    set(GIT_EXECUTABLE ${GIT_EXECUTABLE} CACHE FILEPATH "Path to git executable")
    mark_as_advanced(CLEAR GIT_EXECUTABLE)
    if(BUILD_DISTRIBUTED)
      message(SEND_ERROR "${ColorBoldRed}Git executable was not found, specify GIT_EXECUTABLE as the path to the git executable.${ColorReset}")
    else()
      message(SEND_ERROR "${ColorBoldRed}Git executable was not found. Either specify GIT_EXECUTABLE as the path to the git executable, or use NO_GIT_REVISION to disable.${ColorReset}")
    endif()
  endif()
  set(GIT_HEADER_FILE_TEMPLATE_BARE program/gitinfotemplate.h)
  set(GIT_HEADER_FILE_ALWAYS_UPDATED_BARE program/gitinfoupdated.h)
  set(GIT_HEADER_FILE_BARE program/gitinfo.h)
  set(GIT_HEADER_FILE_TEMPLATE ${CMAKE_SOURCE_DIR}/${GIT_HEADER_FILE_TEMPLATE_BARE})
  set(GIT_HEADER_FILE_ALWAYS_UPDATED ${CMAKE_BINARY_DIR}/${GIT_HEADER_FILE_ALWAYS_UPDATED_BARE})
  set(GIT_HEADER_FILE ${CMAKE_BINARY_DIR}/${GIT_HEADER_FILE_BARE})
  add_custom_command(
    OUTPUT ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    COMMAND ${CMAKE_COMMAND} -E copy ${GIT_HEADER_FILE_TEMPLATE} ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    COMMAND ${GIT_EXECUTABLE} describe --match=DummyTagNotExisting --always --abbrev=40 --dirty >> ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GIT_HEADER_FILE_ALWAYS_UPDATED} ${GIT_HEADER_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
    )
endif()

#--------------------------- KATAGO COMPILING AND LINKING --------------------------------------------------------------

add_executable(katago
  core/global.cpp
  core/base64.cpp
  core/bsearch.cpp
  core/commandloop.cpp
  core/config_parser.cpp
  core/datetime.cpp
  core/elo.cpp
  core/fancymath.cpp
  core/fileutils.cpp
  core/hash.cpp
  core/logger.cpp
  core/mainargs.cpp
  core/makedir.cpp
  core/md5.cpp
  core/multithread.cpp
  core/rand.cpp
  core/rand_helpers.cpp
  core/sha2.cpp
  core/test.cpp
  core/threadsafecounter.cpp
  core/threadsafequeue.cpp
  core/threadtest.cpp
  core/timer.cpp
  game/board.cpp
  game/rules.cpp
  game/gamelogic.cpp
  game/randomopening.cpp
  game/boardhistory.cpp
  game/graphhash.cpp
  dataio/sgf.cpp
  dataio/numpywrite.cpp
  dataio/trainingwrite.cpp
  dataio/loadmodel.cpp
  dataio/homedata.cpp
  dataio/files.cpp
  dataio/poswriter.cpp
  neuralnet/nninputs.cpp
  neuralnet/modelversion.cpp
  neuralnet/int8policy.cpp
  neuralnet/v105policy.cpp
  ${FOUR_PROFILE_PROVIDER_SOURCES}
  neuralnet/nneval.cpp
  neuralnet/desc.cpp
  neuralnet/onnxprotoreader.cpp
  ${NEURALNET_BACKEND_SOURCES}
  book/book.cpp
  book/bookcssjs.cpp
  search/timecontrols.cpp
  search/searchparams.cpp
  search/mutexpool.cpp
  search/search.cpp
  search/searchnode.cpp
  search/searchresults.cpp
  search/searchhelpers.cpp
  search/searchexplorehelpers.cpp
  search/searchmultithreadhelpers.cpp
  search/searchnnhelpers.cpp
  search/searchtimehelpers.cpp
  search/searchupdatehelpers.cpp
  search/asyncbot.cpp
  search/distributiontable.cpp
  search/localpattern.cpp
  search/searchnodetable.cpp
  search/subtreevaluebiastable.cpp
  search/patternbonustable.cpp
  search/analysisdata.cpp
  search/reportedsearchvalues.cpp
  program/gtpconfig.cpp
  program/setup.cpp
  program/playutils.cpp
  program/playsettings.cpp
  program/play.cpp
  program/selfplaymanager.cpp
  ${GIT_HEADER_FILE_ALWAYS_UPDATED}
  tests/testcommon.cpp
  tests/testforbiddenbulk.cpp
  tests/testnnevalcanary.cpp
  distributed/client.cpp
  command/commandline.cpp
  command/analysis.cpp
  command/benchmark.cpp
  command/contribute.cpp
  command/evalsgf.cpp
  command/gatekeeper.cpp
  command/genbook.cpp
  command/gputest.cpp
  command/gtp.cpp
  command/gomprotocol.cpp
  command/match.cpp
  command/matchauto.cpp
  command/misc.cpp
  command/runtests.cpp
  command/sandbox.cpp
  command/selfplay.cpp
  command/distill.cpp
  command/tune.cpp
  forbiddenPoint/ForbiddenPointFinder.cpp
  forbiddenPoint/ForbiddenPointFinderBulk.cpp
  vcfsolver/VCFsolver.cpp
  vcfsolver/VCFHashTable.cpp
  main.cpp
  )

if(KATAGO_BUILD_BENCHMARKNN)
  target_sources(katago PRIVATE command/benchmarknn.cpp)
  target_compile_definitions(katago PRIVATE KATAGO_BUILD_BENCHMARKNN=1)
endif()

if(KATAGO_BUILD_V105_WIRE_CONTRACT)
  target_sources(katago PRIVATE tests/v105_wire_contract.cpp)
  target_compile_definitions(katago PRIVATE KATAGO_BUILD_V105_WIRE_CONTRACT=1)
  add_custom_target(v105_wire_contract
    COMMAND $<TARGET_FILE:katago> testv105wire
    DEPENDS katago
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM)
endif()

if(KATAGO_BUILD_NNRAWGATE)
  if(NOT USE_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "KATAGO_BUILD_NNRAWGATE requires USE_BACKEND=CUDA")
  endif()
  target_sources(katago PRIVATE command/nnrawgate.cpp)
  target_compile_definitions(katago PRIVATE KATAGO_BUILD_NNRAWGATE=1)
endif()

if(USE_BACKEND STREQUAL "CUDA")
  target_compile_definitions(katago PRIVATE USE_CUDA_BACKEND)
  target_compile_definitions(katago PRIVATE CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
  if(KATAGO_P1_SM120_AVAILABLE)
    target_compile_definitions(katago PRIVATE
      KATAGO_ENABLE_P1_SM120_PROVIDER=1
      KATAGO_ENABLE_RENJU15_RMS_SM120=1
      KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120=1)
    target_include_directories(katago PRIVATE "${KATAGO_P1_FA4_ROOT}")
  endif()
  if(KATAGO_P2_SM120_AVAILABLE)
    target_compile_definitions(katago PRIVATE
      KATAGO_ENABLE_P2_SM120_PROVIDER=1)
  endif()
  if(KATAGO_P3_PROVIDER_COMPILED)
    target_compile_definitions(katago PRIVATE KATAGO_P3_PROVIDER_COMPILED=1)
  endif()
  if(KATAGO_P4_PROVIDER_COMPILED)
    target_compile_definitions(katago PRIVATE KATAGO_P4_PROVIDER_COMPILED=1)
  endif()
  find_package(CUDAToolkit REQUIRED)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
  if((NOT CUDNN_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  target_link_libraries(katago CUDA::cublas ${CUDNN_LIBRARY})
  if(KATAGO_P1_SM120_AVAILABLE)
    target_link_libraries(katago
      "${KATAGO_P1_FA4_OBJECT}"
      "${KATAGO_P1_FA4_RUNTIME}")
  endif()
  if(KATAGO_P2_SM120_AVAILABLE)
    target_link_libraries(katago
      "${KATAGO_C256_S361_FA4_OBJECT}"
      "${KATAGO_C256_S361_FA4_RUNTIME}")
  endif()
  if(KATAGO_P3_EXTERNAL_LINK_LIBRARIES)
    target_link_libraries(katago ${KATAGO_P3_EXTERNAL_LINK_LIBRARIES})
  endif()
elseif(USE_BACKEND STREQUAL "TENSORRT")
  target_compile_definitions(katago PRIVATE USE_TENSORRT_BACKEND)
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
  if((NOT TENSORRT_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} NvInfer.h was NOT found, specify TENSORRT_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()
  find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" tensorrt_version_header)

  set(TENSORRT_VERSION_MAJOR "")
  set(TENSORRT_VERSION_MINOR "")
  set(TENSORRT_VERSION_PATCH "")

  string(REGEX MATCH "#define NV_TENSORRT_MAJOR +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_MAJOR_ENTERPRISE +([0-9]+)" tensorrt_major_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
    endif()
  endif()

  string(REGEX MATCH "#define NV_TENSORRT_MINOR +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_MINOR_ENTERPRISE +([0-9]+)" tensorrt_minor_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
  endif()

  string(REGEX MATCH "#define NV_TENSORRT_PATCH +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_PATCH_ENTERPRISE +([0-9]+)" tensorrt_patch_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
    endif()
  endif()

  set(TENSORRT_VERSION "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}")
  message(STATUS "Detected TensorRT version: ${TENSORRT_VERSION}")

  if("${TENSORRT_VERSION_MAJOR}" STREQUAL "" OR "${TENSORRT_VERSION_MINOR}" STREQUAL "" OR "${TENSORRT_VERSION_PATCH}" STREQUAL "")
    message(FATAL_ERROR "Could not determine TensorRT version from header file")
  endif()

  # Version 8 is required for serializing the builder timing cache.
  # Version 8.2 is required for eliminating the global logger for Builder and Runtime.
  # Version 8.5 is required for eliminating many deprecated APIs and adopting new features.
  # Version 8.6 is for CUDA 12 support and further reduction in initialization time.
  if(TENSORRT_VERSION VERSION_LESS 8.6)
    message(FATAL_ERROR "TensorRT 8.6 or greater is required but ${TENSORRT_VERSION} was found.")
  endif()
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  target_link_libraries(katago CUDA::cudart_static ${TENSORRT_LIBRARY})  
  find_library(TENSORRT_ONNX_LIBRARY nvonnxparser HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES lib)
  target_link_libraries(katago ${TENSORRT_ONNX_LIBRARY})
elseif(USE_BACKEND STREQUAL "OPENCL")
  target_compile_definitions(katago PRIVATE USE_OPENCL_BACKEND)
  find_package(OpenCL)
  if(NOT OpenCL_FOUND)
    message(WARNING "OpenCL not found, attempting to see if CUDA exists and has OpenCL since sometimes CUDA may provide OpenCL where cmake can't find it.")
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR "OpenCL installation not found")
    else()
      message(WARNING "OpenCL not found, but found CUDA, attempting to use OpenCL via CUDA.")
    endif()
    include_directories(${OpenCL_INCLUDE_DIRS})
    include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(katago CUDA::OpenCL)
  else()
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
    target_link_libraries(katago ${OpenCL_LIBRARY})
  endif()
elseif(USE_BACKEND STREQUAL "EIGEN")
  target_compile_definitions(katago PRIVATE USE_EIGEN_BACKEND)
  if(NOT (MSVC OR MINGW))
    find_package(Eigen3 REQUIRED)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIRS}")
  else()
    if(EIGEN3_INCLUDE_DIRS)
      message(STATUS "EIGEN3_INCLUDE_DIRS is set to ${EIGEN3_INCLUDE_DIRS}, assuming that Eigen3 header files are here.")
      include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
    else()
      find_package(Eigen3)
      if(NOT Eigen3_FOUND)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS} CACHE PATH "Directory containing 'Eigen' and 'unsupported' subdirs with Eigen headers")
        message(FATAL_ERROR "*** KataGo: Eigen3 was not found. If you've downloaded Eigen3, you can IGNORE the whole above error from CMake and just set 'EIGEN3_INCLUDE_DIRS' to the root eigen3-3.*.* directory after unzipping it. Among other things it should contain 'Eigen' and 'unsupported' subdirs, which is all KataGo needs. Otherwise, you will need to 'build' Eigen3 even though it is header-only and install it via MSVC in order for cmake to find it via FindEigen3.cmake.")
      else()
        include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
        message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIRS}")
      endif()
    endif()
  endif()
elseif(USE_BACKEND STREQUAL "ONNX-CPU")
  target_compile_definitions(katago PRIVATE USE_ONNX_CPU_BACKEND)

  set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime installation")

  find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_ROOT} ${ONNXRUNTIME_ROOT}/include PATH_SUFFIXES include core/session)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime HINTS ${ONNXRUNTIME_ROOT} ${ONNXRUNTIME_ROOT}/lib PATH_SUFFIXES lib)

  if(NOT ONNXRUNTIME_INCLUDE_DIR)
     set(ONNXRUNTIME_INCLUDE_DIR "ONNXRUNTIME_INCLUDE_DIR-NOTFOUND" CACHE PATH "Path to ONNX Runtime include directory")
     message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR not found. Please set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory, or set ONNXRUNTIME_INCLUDE_DIR manually.")
  endif()
  if(NOT ONNXRUNTIME_LIBRARY)
     set(ONNXRUNTIME_LIBRARY "ONNXRUNTIME_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to ONNX Runtime library")
     message(FATAL_ERROR "ONNXRUNTIME_LIBRARY not found. Please set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory, or set ONNXRUNTIME_LIBRARY manually.")
  endif()

  include_directories(${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(katago ${ONNXRUNTIME_LIBRARY})

elseif(USE_BACKEND STREQUAL "ONNX-DIRECTML")
  target_compile_definitions(katago PRIVATE USE_ONNX_DIRECTML_BACKEND)
  
endif()

if(USE_BIGGER_BOARDS_EXPENSIVE)
  target_compile_definitions(katago PRIVATE COMPILE_MAX_BOARD_LEN=29)
endif()

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
endif()

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  target_link_libraries(katago ${ZLIB_LIBRARIES})
else()
  set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIR} CACHE PATH "Path to directory with zlib.h and other header files")
  set(ZLIB_LIBRARY ${ZLIB_LIBRARY} CACHE FILEPATH "Path to 'libz.so' on Linux or 'libz.lib' on Windows")
  mark_as_advanced(CLEAR ZLIB_INCLUDE_DIR ZLIB_LIBRARY)
  message(SEND_ERROR "${ColorBoldRed}zlib was not found, if zlib is actually installed but not being found you can set ZLIB_INCLUDE_DIR to the directory with zlib.h and other headers, and ZLIB_LIBRARY to the compiled library 'libz.so' on Linux or 'libz.lib' on Windows. On the command line, this is -DZLIB_INCLUDE_DIR=... and -DZLIB_LIBRARY=... ${ColorReset}")
endif(ZLIB_FOUND)

find_library(LIBZIP_LIBRARY NAMES zip)
find_path(LIBZIP_INCLUDE_DIR_ZIP NAMES zip.h)
find_path(LIBZIP_INCLUDE_DIR_ZIPCONF NAMES zipconf.h)
if((NOT LIBZIP_LIBRARY) OR (NOT LIBZIP_INCLUDE_DIR_ZIP) OR (NOT LIBZIP_INCLUDE_DIR_ZIPCONF))
  if(BUILD_DISTRIBUTED)
    message(SEND_ERROR "${ColorBoldRed}WARNING: BUILD_DISTRIBUTED was requested but libzip library was NOT found. KataGo needs this for writing training data so libzip is required. On Linux, install through your normal package manager. On Windows, set LIBZIP_INCLUDE_DIR_ZIP to the directory that includes zip.h and other files, and LIBZIP_INCLUDE_DIR_ZIPCONF to the directory that includes zipconf.h and other files, and LIBZIP_LIBRARY to the libzip.lib or zip.lib file. ${ColorReset}")
  endif()
  target_compile_definitions(katago PRIVATE NO_LIBZIP)
  message(WARNING "${ColorBoldRed}WARNING: libzip library was NOT found. KataGo should still work for GTP/matches/analysis if everything else is good, but selfplay for writing training data will not be possible.${ColorReset}")
  set(LIBZIP_INCLUDE_DIR_ZIP ${LIBZIP_INCLUDE_DIR_ZIP} CACHE PATH "Path to directory with zip.h and other header files")
  set(LIBZIP_INCLUDE_DIR_ZIPCONF ${LIBZIP_INCLUDE_DIR_ZIPCONF} CACHE PATH "Path to directory with zipconf.h and other header files")
  set(LIBZIP_LIBRARY ${LIBZIP_LIBRARY} CACHE FILEPATH "Path to 'libzip.so' on Linux or 'libzip.lib' or 'zip.lib' on Windows")
  mark_as_advanced(CLEAR LIBZIP_INCLUDE_DIR_ZIP LIBZIP_INCLUDE_DIR_ZIPCONF LIBZIP_LIBRARY)
else()
  include_directories(${LIBZIP_INCLUDE_DIR_ZIP})
  include_directories(${LIBZIP_INCLUDE_DIR_ZIPCONF})
  target_link_libraries(katago ${LIBZIP_LIBRARY})
endif()

if(USE_TCMALLOC)
  target_link_libraries(katago ${TCMALLOC_LIB})
endif(USE_TCMALLOC)

if(BUILD_DISTRIBUTED)
  message(STATUS "-DBUILD_DISTRIBUTED=1 is set, compiling code and dependencies to contribute to distributed training")
  target_compile_definitions(katago PRIVATE BUILD_DISTRIBUTED)
  find_package(OpenSSL REQUIRED)
  target_link_libraries(katago ${OPENSSL_SSL_LIBRARIES} ${OPENSSL_CRYPTO_LIBRARIES})
  include_directories(${OPENSSL_INCLUDE_DIR})
  include_directories(external/httplib)
endif()

#------------------------------------------------------------------------------------

# add_compile_definitions(NDEBUG)

if(MSVC)
  message(STATUS "Setting up build for MSVC.")
  # Suppress min and max macros on windows
  # Also define a few other things for windows
  target_compile_definitions(katago PRIVATE NOMINMAX)
  target_compile_definitions(katago PRIVATE BYTE_ORDER=1234)
  target_compile_definitions(katago PRIVATE LITTLE_ENDIAN=1234)
  target_compile_definitions(katago PRIVATE BIG_ENDIAN=4321)
  # core/rand.cpp uses winsock for a gethostname
  target_link_libraries(katago ws2_32)

  # Go ahead and suppress some MSVC warnings about sprintf and similar
  # things. They might be useful in some cases, but also are noisy.
  target_compile_definitions(katago PRIVATE _CRT_SECURE_NO_WARNINGS)

  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:AVX2 -D__FMA__")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8388608")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "Setting up build for GNU or Clang.")
  if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(arm|aarch32|aarch64)"))
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse")
  endif()
  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  find_package (Threads REQUIRED)
  target_link_libraries(katago Threads::Threads)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Wmissing-declarations -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-overflow=1 -Wswitch-default -Wfloat-conversion -Wunused")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Enabling GNU-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-op -Wnoexcept -Wstrict-null-sentinel")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6.1)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Walloc-zero -Wduplicated-branches -Wduplicated-cond -Wdangling-else -Wrestrict")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Enabling AppleClang-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Wdangling-else")
  else()
    message(STATUS "Enabling Clang-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Wdangling-else")
    target_link_libraries(katago "atomic")
  endif()

  if(USE_TCMALLOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
  endif()

endif()

target_include_directories(katago PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

