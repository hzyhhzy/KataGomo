cmake_minimum_required(VERSION 3.18.2)
project(katago)

set(CMAKE_CXX_STANDARD 17)

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
set(KATAGO_ENABLE_CUDNN_FRONTEND_SDPA 1 CACHE BOOL
  "Enable the vendored cuDNN Frontend FP16 scaled-dot-product-attention path in the CUDA backend")
set(KATAGO_ENABLE_SM120_TRANSFORMER_WINNER 0 CACHE BOOL
  "Build selected SM120 transformer tactics when CUDA 13 and pinned CUTLASS are available")
set(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  ${KATAGO_ENABLE_SM120_TRANSFORMER_WINNER} CACHE BOOL
  "Build qualified clip4/per-matrix INT8 C256 QK and FFN tactics with the SM120 winner")
set(KATAGO_CUTLASS_ROOT "" CACHE PATH
  "NVIDIA CUTLASS checkout pinned to dcf215af68a2d08d305076c152a06f201728cd53")
set(KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST "" CACHE FILEPATH
  "Optional generated CMake manifest containing audited C384 exact-AOT bridges and external objects")
set(KATAGO_C384_EXACT_QKV_TACTIC_ID "" CACHE STRING
  "Exact generated QKV+RoPE candidate ID selected at construction; empty disables the family")
set(KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID "" CACHE STRING
  "Exact generated dual-FFN candidate ID selected at construction; empty disables the family")
set(KATAGO_C384_H12_FA4_PACKAGE_CMAKE "" CACHE FILEPATH
  "Optional SHA-locked C384/H12 exact-FA4 packed-token package")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/C384ExactAotPolicy.cmake")
katago_c384_exact_validate_request_policy()
set(KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID "")
set(KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID "")

#--------------------------- NEURAL NET BACKEND ------------------------------------------------------------------------

message(STATUS "Building 'katago' executable for GTP engine and other tools.")
if(USE_BACKEND STREQUAL "CUDA")
  message(STATUS "-DUSE_BACKEND=CUDA, using CUDA backend.")

  enable_language(CUDA)
  katago_c384_exact_validate_cuda_version()

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/cudabackend.cpp
    neuralnet/cudabackend_qkv_planar.cpp
    neuralnet/cudautils.cpp
    neuralnet/cudahelpers.cu
  )

  if(KATAGO_ENABLE_SM120_TRANSFORMER_WINNER)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(STATUS
        "SM120 transformer winner disabled: CUDA 13.0 or newer is required")
      set(KATAGO_ENABLE_SM120_TRANSFORMER_WINNER 0)
      set(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT 0)
    endif()
  endif()

  if((NOT KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST STREQUAL "" OR
      NOT KATAGO_C384_H12_FA4_PACKAGE_CMAKE STREQUAL "") AND
     NOT KATAGO_ENABLE_SM120_TRANSFORMER_WINNER)
    message(FATAL_ERROR
      "C384 exact-AOT/FA4 packages require the SM120 winner and CUDA 13+")
  endif()

  foreach(KATAGO_C384_EXACT_ID IN ITEMS
      "${KATAGO_C384_EXACT_QKV_TACTIC_ID}"
      "${KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID}")
    if(NOT KATAGO_C384_EXACT_ID STREQUAL "" AND
       NOT KATAGO_C384_EXACT_ID MATCHES "^[a-z0-9][a-z0-9-]*$")
      message(FATAL_ERROR "Unsafe C384 exact-AOT candidate ID: ${KATAGO_C384_EXACT_ID}")
    endif()
  endforeach()
  if(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT AND
     NOT KATAGO_ENABLE_SM120_TRANSFORMER_WINNER)
    message(FATAL_ERROR
      "KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT requires the SM120 transformer winner")
  endif()

  if(KATAGO_ENABLE_SM120_TRANSFORMER_WINNER)
    set(KATAGO_PINNED_CUTLASS_COMMIT
      "dcf215af68a2d08d305076c152a06f201728cd53")
    if(NOT EXISTS "${KATAGO_CUTLASS_ROOT}/include/cutlass/cutlass.h" OR
       NOT EXISTS "${KATAGO_CUTLASS_ROOT}/examples/45_dual_gemm/device/dual_gemm.h")
      message(FATAL_ERROR
        "KATAGO_CUTLASS_ROOT must point to CUTLASS ${KATAGO_PINNED_CUTLASS_COMMIT}")
    endif()
    find_package(Git REQUIRED)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${KATAGO_CUTLASS_ROOT}" rev-parse HEAD
      RESULT_VARIABLE KATAGO_CUTLASS_REVISION_RESULT
      OUTPUT_VARIABLE KATAGO_CUTLASS_REVISION
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT KATAGO_CUTLASS_REVISION_RESULT EQUAL 0 OR
       NOT KATAGO_CUTLASS_REVISION STREQUAL KATAGO_PINNED_CUTLASS_COMMIT)
      message(FATAL_ERROR
        "CUTLASS revision is '${KATAGO_CUTLASS_REVISION}', expected '${KATAGO_PINNED_CUTLASS_COMMIT}'")
    endif()

    set(KATAGO_RENJU15_FA4_B36_AOT_ROOT
      "${CMAKE_CURRENT_SOURCE_DIR}/external/renju15-sm120-fa4")
    set(KATAGO_RENJU15_FA4_B36_HEADER
      "${KATAGO_RENJU15_FA4_B36_AOT_ROOT}/renju15_fa4_b36_tn128.h")
    set(KATAGO_RENJU15_FA4_B36_OBJECT
      "${KATAGO_RENJU15_FA4_B36_AOT_ROOT}/renju15_fa4_b36_tn128.o")
    set(KATAGO_RENJU15_FA4_B36_METADATA
      "${KATAGO_RENJU15_FA4_B36_AOT_ROOT}/renju15_fa4_b36_tn128.json")
    set(KATAGO_RENJU15_FA4_CUTE_RUNTIME
      "${KATAGO_RENJU15_FA4_B36_AOT_ROOT}/libcuda_dialect_runtime_static.a")
    set(KATAGO_RENJU15_FA4_ARTIFACTS
      "${KATAGO_RENJU15_FA4_B36_HEADER}"
      "${KATAGO_RENJU15_FA4_B36_OBJECT}"
      "${KATAGO_RENJU15_FA4_B36_METADATA}"
      "${KATAGO_RENJU15_FA4_CUTE_RUNTIME}")
    foreach(KATAGO_WINNER_ARTIFACT IN LISTS KATAGO_RENJU15_FA4_ARTIFACTS)
      if(NOT EXISTS "${KATAGO_WINNER_ARTIFACT}")
        message(FATAL_ERROR "Missing bundled SM120 winner artifact: ${KATAGO_WINNER_ARTIFACT}")
      endif()
    endforeach()
    file(SHA256 "${KATAGO_RENJU15_FA4_B36_HEADER}" KATAGO_RENJU15_FA4_HEADER_SHA)
    file(SHA256 "${KATAGO_RENJU15_FA4_B36_OBJECT}" KATAGO_RENJU15_FA4_OBJECT_SHA)
    file(SHA256 "${KATAGO_RENJU15_FA4_B36_METADATA}" KATAGO_RENJU15_FA4_METADATA_SHA)
    file(SHA256 "${KATAGO_RENJU15_FA4_CUTE_RUNTIME}" KATAGO_RENJU15_FA4_RUNTIME_SHA)
    if(NOT KATAGO_RENJU15_FA4_HEADER_SHA STREQUAL
         "48e8572e7dda80f7022085e0384830e88f3a4f58e6a79f25c40b0ff60167311d" OR
       NOT KATAGO_RENJU15_FA4_OBJECT_SHA STREQUAL
         "690fbd68a750e90ad4ea964ecb729258ccce8d5d81a8e996dc436435dcb62a9d" OR
       NOT KATAGO_RENJU15_FA4_METADATA_SHA STREQUAL
         "08c8b0121d15c75c47dcc6dde987eff4dba64840989a301091063c57e518e925" OR
       NOT KATAGO_RENJU15_FA4_RUNTIME_SHA STREQUAL
         "5ffeb7ae24bf02755a6ad751d196b2f1efd18a834b747972ffdd8bbc6550b899")
      message(FATAL_ERROR
        "Bundled SM120 FA4 artifacts failed the clean generation SHA256 contract")
    endif()

    set(KATAGO_C384_H12_FA4_ENABLED 0)
    set(KATAGO_C384_H12_FA4_BRIDGES "")
    set(KATAGO_C384_H12_FA4_OBJECTS "")
    set(KATAGO_C384_H12_FA4_INCLUDE_DIRS "")
    set(KATAGO_C384_H12_FA4_RUNTIME "")
    if(KATAGO_C384_H12_FA4_PACKAGE_CMAKE STREQUAL "")
      set(KATAGO_C384_H12_FA4_REGISTRY_PROVIDER
        neuralnet/c384_h12_fa4_registry_stub.cpp)
    else()
      get_filename_component(KATAGO_C384_H12_FA4_PACKAGE_CMAKE
        "${KATAGO_C384_H12_FA4_PACKAGE_CMAKE}" ABSOLUTE)
      if(NOT EXISTS "${KATAGO_C384_H12_FA4_PACKAGE_CMAKE}")
        message(FATAL_ERROR
          "KATAGO_C384_H12_FA4_PACKAGE_CMAKE does not exist: ${KATAGO_C384_H12_FA4_PACKAGE_CMAKE}")
      endif()
      include("${KATAGO_C384_H12_FA4_PACKAGE_CMAKE}")
      if(NOT KATAGO_C384_H12_FA4_PACKAGE_SCHEMA STREQUAL "1")
        message(FATAL_ERROR "C384/H12 FA4 package schema must be 1")
      endif()
      if(NOT KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "BATCH_SEARCH" AND
         NOT KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "PRODUCTION")
        message(FATAL_ERROR
          "C384/H12 FA4 package mode must be BATCH_SEARCH or PRODUCTION")
      endif()
      list(LENGTH KATAGO_C384_H12_FA4_BATCHES KATAGO_C384_H12_FA4_BATCH_COUNT)
      set(KATAGO_C384_H12_FA4_SORTED_BATCHES ${KATAGO_C384_H12_FA4_BATCHES})
      list(SORT KATAGO_C384_H12_FA4_SORTED_BATCHES COMPARE NATURAL ORDER ASCENDING)
      if(KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "BATCH_SEARCH")
        if(NOT KATAGO_C384_H12_FA4_BATCH_COUNT EQUAL 2 OR
           NOT KATAGO_C384_H12_FA4_SORTED_BATCHES STREQUAL "24;28")
          message(FATAL_ERROR
            "C384/H12 FA4 batch search must contain exactly B24 and B28")
        endif()
      elseif(NOT KATAGO_C384_H12_FA4_BATCH_COUNT EQUAL 1 OR
             (NOT KATAGO_C384_H12_FA4_SORTED_BATCHES STREQUAL "24" AND
              NOT KATAGO_C384_H12_FA4_SORTED_BATCHES STREQUAL "28"))
        message(FATAL_ERROR
          "C384/H12 FA4 production package must collapse to B24 or B28")
      endif()
      foreach(KATAGO_C384_H12_FA4_LIST HEADERS OBJECTS BRIDGES METADATA)
        list(LENGTH KATAGO_C384_H12_FA4_${KATAGO_C384_H12_FA4_LIST}
          KATAGO_C384_H12_FA4_${KATAGO_C384_H12_FA4_LIST}_COUNT)
        if(NOT KATAGO_C384_H12_FA4_${KATAGO_C384_H12_FA4_LIST}_COUNT EQUAL
               KATAGO_C384_H12_FA4_BATCH_COUNT)
          message(FATAL_ERROR
            "C384/H12 FA4 ${KATAGO_C384_H12_FA4_LIST} count does not match batches")
        endif()
      endforeach()
      list(LENGTH KATAGO_C384_H12_FA4_ASSETS KATAGO_C384_H12_FA4_ASSET_COUNT)
      list(LENGTH KATAGO_C384_H12_FA4_ASSET_SHAS KATAGO_C384_H12_FA4_SHA_COUNT)
      if(NOT KATAGO_C384_H12_FA4_ASSET_COUNT EQUAL KATAGO_C384_H12_FA4_SHA_COUNT OR
         KATAGO_C384_H12_FA4_ASSET_COUNT EQUAL 0)
        message(FATAL_ERROR "C384/H12 FA4 package asset/SHA lists are invalid")
      endif()
      set(KATAGO_C384_H12_FA4_EXPECTED_ASSETS
        ${KATAGO_C384_H12_FA4_HEADERS}
        ${KATAGO_C384_H12_FA4_OBJECTS}
        ${KATAGO_C384_H12_FA4_BRIDGES}
        ${KATAGO_C384_H12_FA4_METADATA})
      set(KATAGO_C384_H12_FA4_SORTED_ASSETS ${KATAGO_C384_H12_FA4_ASSETS})
      list(SORT KATAGO_C384_H12_FA4_SORTED_ASSETS)
      list(SORT KATAGO_C384_H12_FA4_EXPECTED_ASSETS)
      if(NOT KATAGO_C384_H12_FA4_SORTED_ASSETS STREQUAL
             KATAGO_C384_H12_FA4_EXPECTED_ASSETS)
        message(FATAL_ERROR
          "C384/H12 FA4 ASSETS must equal headers+objects+bridges+metadata")
      endif()
      math(EXPR KATAGO_C384_H12_FA4_LAST_ASSET
        "${KATAGO_C384_H12_FA4_ASSET_COUNT} - 1")
      foreach(KATAGO_C384_H12_FA4_INDEX RANGE 0 ${KATAGO_C384_H12_FA4_LAST_ASSET})
        list(GET KATAGO_C384_H12_FA4_ASSETS ${KATAGO_C384_H12_FA4_INDEX}
          KATAGO_C384_H12_FA4_ASSET)
        list(GET KATAGO_C384_H12_FA4_ASSET_SHAS ${KATAGO_C384_H12_FA4_INDEX}
          KATAGO_C384_H12_FA4_EXPECTED_SHA)
        if(NOT EXISTS "${KATAGO_C384_H12_FA4_ASSET}")
          message(FATAL_ERROR "Missing C384/H12 FA4 asset: ${KATAGO_C384_H12_FA4_ASSET}")
        endif()
        file(SHA256 "${KATAGO_C384_H12_FA4_ASSET}"
          KATAGO_C384_H12_FA4_ACTUAL_SHA)
        if(KATAGO_C384_H12_FA4_EXPECTED_SHA STREQUAL "" OR
           NOT KATAGO_C384_H12_FA4_ACTUAL_SHA STREQUAL
             KATAGO_C384_H12_FA4_EXPECTED_SHA)
          message(FATAL_ERROR
            "C384/H12 FA4 asset SHA256 mismatch: ${KATAGO_C384_H12_FA4_ASSET}")
        endif()
      endforeach()
      foreach(KATAGO_C384_H12_FA4_SINGLE REGISTRY RUNTIME)
        if(NOT EXISTS "${KATAGO_C384_H12_FA4_${KATAGO_C384_H12_FA4_SINGLE}}")
          message(FATAL_ERROR
            "Missing C384/H12 FA4 ${KATAGO_C384_H12_FA4_SINGLE}")
        endif()
        file(SHA256 "${KATAGO_C384_H12_FA4_${KATAGO_C384_H12_FA4_SINGLE}}"
          KATAGO_C384_H12_FA4_SINGLE_ACTUAL_SHA)
        if(NOT KATAGO_C384_H12_FA4_SINGLE_ACTUAL_SHA STREQUAL
             KATAGO_C384_H12_FA4_${KATAGO_C384_H12_FA4_SINGLE}_SHA)
          message(FATAL_ERROR
            "C384/H12 FA4 ${KATAGO_C384_H12_FA4_SINGLE} SHA256 mismatch")
        endif()
      endforeach()
      set(KATAGO_C384_H12_FA4_REGISTRY_PROVIDER
        "${KATAGO_C384_H12_FA4_REGISTRY}")
      foreach(KATAGO_C384_H12_FA4_HEADER IN LISTS KATAGO_C384_H12_FA4_HEADERS)
        get_filename_component(KATAGO_C384_H12_FA4_HEADER_DIR
          "${KATAGO_C384_H12_FA4_HEADER}" DIRECTORY)
        list(APPEND KATAGO_C384_H12_FA4_INCLUDE_DIRS
          "${KATAGO_C384_H12_FA4_HEADER_DIR}")
      endforeach()
      list(REMOVE_DUPLICATES KATAGO_C384_H12_FA4_INCLUDE_DIRS)
      set(KATAGO_C384_H12_FA4_ENABLED 1)
      message(STATUS
        "C384/H12 FA4 package mode=${KATAGO_C384_H12_FA4_PACKAGE_MODE} batches=${KATAGO_C384_H12_FA4_BATCHES} registry_sha=${KATAGO_C384_H12_FA4_REGISTRY_SHA}")
    endif()

    # The ordinary backend stays a fat CUDA build. Only the source-specialized
    # winner kernels form this SM120 object island.
    foreach(KATAGO_C384_EXACT_AOT_MANIFEST_VALUE IN ITEMS
        KATAGO_C384_EXACT_AOT_PACKAGE_SCHEMA
        KATAGO_C384_EXACT_AOT_PACKAGE_MODE
        KATAGO_C384_EXACT_AOT_SELECTED_BATCH
        KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID
        KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID
        KATAGO_C384_EXACT_AOT_SELECTED_FAMILIES
        KATAGO_C384_EXACT_AOT_QKV_ROPE_IDS
        KATAGO_C384_EXACT_AOT_DUAL_FFN_IDS
        KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE
        KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE_SHA256
        KATAGO_C384_EXACT_AOT_REGISTRY_PROVIDER
        KATAGO_C384_EXACT_AOT_GENERATED_INCLUDE_DIR
        KATAGO_C384_EXACT_AOT_GENERATED_HEADERS
        KATAGO_C384_EXACT_AOT_GENERATED_METADATA
        KATAGO_C384_EXACT_AOT_GENERATED_BRIDGES
        KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS
        KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256)
      set(${KATAGO_C384_EXACT_AOT_MANIFEST_VALUE} "")
    endforeach()
    if(NOT KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST STREQUAL "")
      # Preserve optional user expectations before including generated CMake.
      # The manifest selects the effective IDs; it may not rewrite the values
      # against which it is checked.
      set(KATAGO_C384_EXACT_REQUESTED_QKV_TACTIC_ID
        "${KATAGO_C384_EXACT_QKV_TACTIC_ID}")
      set(KATAGO_C384_EXACT_REQUESTED_DUAL_FFN_TACTIC_ID
        "${KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID}")
      get_filename_component(KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST
        "${KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST}" ABSOLUTE)
      if(NOT EXISTS "${KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST}")
        message(FATAL_ERROR
          "KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST does not exist: ${KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST}")
      endif()
      include("${KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST}")
      katago_c384_exact_resolve_manifest_policy()
      foreach(KATAGO_C384_EXACT_AOT_FILE IN LISTS
          KATAGO_C384_EXACT_AOT_GENERATED_HEADERS
          KATAGO_C384_EXACT_AOT_GENERATED_METADATA
          KATAGO_C384_EXACT_AOT_GENERATED_BRIDGES
          KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS)
        if(NOT EXISTS "${KATAGO_C384_EXACT_AOT_FILE}")
          message(FATAL_ERROR
            "Generated C384 exact-AOT file does not exist: ${KATAGO_C384_EXACT_AOT_FILE}")
        endif()
      endforeach()
      if(NOT EXISTS "${KATAGO_C384_EXACT_AOT_REGISTRY_PROVIDER}" OR
         NOT IS_DIRECTORY "${KATAGO_C384_EXACT_AOT_GENERATED_INCLUDE_DIR}")
        message(FATAL_ERROR "Generated C384 exact-AOT manifest is incomplete")
      endif()
      list(LENGTH KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256
        KATAGO_C384_EXACT_AOT_HASH_LIST_LENGTH)
      math(EXPR KATAGO_C384_EXACT_AOT_HASH_LIST_REMAINDER
        "${KATAGO_C384_EXACT_AOT_HASH_LIST_LENGTH} % 2")
      if(KATAGO_C384_EXACT_AOT_HASH_LIST_LENGTH EQUAL 0 OR
         NOT KATAGO_C384_EXACT_AOT_HASH_LIST_REMAINDER EQUAL 0)
        message(FATAL_ERROR "Generated C384 exact-AOT SHA256 list is malformed")
      endif()
      math(EXPR KATAGO_C384_EXACT_AOT_HASH_LIST_LAST
        "${KATAGO_C384_EXACT_AOT_HASH_LIST_LENGTH} - 2")
      foreach(KATAGO_C384_EXACT_AOT_HASH_INDEX RANGE
          0 ${KATAGO_C384_EXACT_AOT_HASH_LIST_LAST} 2)
        math(EXPR KATAGO_C384_EXACT_AOT_HASH_VALUE_INDEX
          "${KATAGO_C384_EXACT_AOT_HASH_INDEX} + 1")
        list(GET KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256
          ${KATAGO_C384_EXACT_AOT_HASH_INDEX} KATAGO_C384_EXACT_AOT_HASH_FILE)
        list(GET KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256
          ${KATAGO_C384_EXACT_AOT_HASH_VALUE_INDEX} KATAGO_C384_EXACT_AOT_HASH_WANTED)
        if(NOT EXISTS "${KATAGO_C384_EXACT_AOT_HASH_FILE}")
          message(FATAL_ERROR
            "Generated C384 exact-AOT hashed file is missing: ${KATAGO_C384_EXACT_AOT_HASH_FILE}")
        endif()
        file(SHA256 "${KATAGO_C384_EXACT_AOT_HASH_FILE}"
          KATAGO_C384_EXACT_AOT_HASH_ACTUAL)
        if(NOT KATAGO_C384_EXACT_AOT_HASH_ACTUAL STREQUAL
           KATAGO_C384_EXACT_AOT_HASH_WANTED)
          message(FATAL_ERROR
            "Generated C384 exact-AOT SHA256 mismatch: ${KATAGO_C384_EXACT_AOT_HASH_FILE}")
        endif()
      endforeach()
    else()
      set(KATAGO_C384_EXACT_AOT_REGISTRY_PROVIDER
        neuralnet/c384_exact_fixed_aot_registry_stub.cu)
    endif()

    set(KATAGO_SM120_TRANSFORMER_WINNER_SOURCES
      neuralnet/cudabackend_sm120_renju15_kernels.cu
      neuralnet/renju15_dual_ffn_sm120.cu
      neuralnet/renju15_residual_gemm_sm120.cu
      neuralnet/renju15_qkv_rope_gemm_sm120.cu
      neuralnet/c384_exact_fixed_aot_registry.cu
      ${KATAGO_C384_EXACT_AOT_REGISTRY_PROVIDER}
      ${KATAGO_C384_EXACT_AOT_GENERATED_BRIDGES}
      ${KATAGO_C384_H12_FA4_BRIDGES})
    if(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT)
      list(APPEND KATAGO_SM120_TRANSFORMER_WINNER_SOURCES
        neuralnet/renju15_int8_fused_sm120.cu)
    endif()
    add_library(katago_sm120_transformer_winner OBJECT
      ${KATAGO_SM120_TRANSFORMER_WINNER_SOURCES})
    set_target_properties(katago_sm120_transformer_winner PROPERTIES
      CUDA_ARCHITECTURES "120"
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON)
    target_compile_options(katago_sm120_transformer_winner PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    target_compile_definitions(katago_sm120_transformer_winner PRIVATE
      KATAGO_ENABLE_RENJU15_RMS_SM120=1
      KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120=1
      KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120=1
      KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120=1
      KATAGO_ENABLE_C384_EXACT_FIXED_AOT_SEARCH=1)
    if(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT)
      target_compile_definitions(katago_sm120_transformer_winner PRIVATE
        KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT=1)
    endif()
    target_include_directories(katago_sm120_transformer_winner PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT KATAGO_C384_EXACT_AOT_GENERATED_INCLUDE_DIR STREQUAL "")
      target_include_directories(katago_sm120_transformer_winner PRIVATE
        "${KATAGO_C384_EXACT_AOT_GENERATED_INCLUDE_DIR}")
    endif()
    if(KATAGO_C384_H12_FA4_INCLUDE_DIRS)
      target_include_directories(katago_sm120_transformer_winner PRIVATE
        ${KATAGO_C384_H12_FA4_INCLUDE_DIRS})
    endif()
    target_include_directories(katago_sm120_transformer_winner SYSTEM PRIVATE
      "${KATAGO_CUTLASS_ROOT}/include"
      "${KATAGO_CUTLASS_ROOT}/examples/45_dual_gemm")
    if(KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS)
      set_source_files_properties(${KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS}
        PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
      target_sources(katago_sm120_transformer_winner PRIVATE
        ${KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS})
    endif()
    if(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT)
      add_executable(katago_renju15_int8_weight_gate EXCLUDE_FROM_ALL
        neuralnet/renju15_int8_weight_gate.cpp)
      add_executable(katago_renju15_int8_rms_gate EXCLUDE_FROM_ALL
        neuralnet/renju15_int8_rms_gate.cu
        neuralnet/cudabackend_sm120_renju15_kernels.cu)
      set_target_properties(katago_renju15_int8_rms_gate PROPERTIES
        CUDA_ARCHITECTURES "120"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
      target_compile_definitions(katago_renju15_int8_rms_gate PRIVATE
        KATAGO_ENABLE_RENJU15_RMS_SM120=1
        KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT=1)
    endif()
    list(APPEND NEURALNET_BACKEND_SOURCES
      neuralnet/renju15_fa4_sm120.cpp
      neuralnet/c384_h12_fa4_sm120.cpp
      ${KATAGO_C384_H12_FA4_REGISTRY_PROVIDER}
      $<TARGET_OBJECTS:katago_sm120_transformer_winner>)
  endif()
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
  set(KATAGO_CUDA_ARCHITECTURES "" CACHE STRING "Override CUDA architectures (for example: 120 for RTX 5090)")
  if(NOT KATAGO_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES ${KATAGO_CUDA_ARCHITECTURES})
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
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
  neuralnet/nneval.cpp
  neuralnet/int8policy.cpp
  neuralnet/desc.cpp
  neuralnet/nativeint8quant.cpp
  neuralnet/architecturedesc.cpp
  neuralnet/cudaopregistry.cpp
  neuralnet/c384_exact_fixed_aot_plan.cpp
  neuralnet/c384_exact_fixed_aot_weights.cpp
  neuralnet/cudabackend_transformer_winner.cpp
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
  tests/testarchitecturedesc.cpp
  tests/testtransformerproductionplan.cpp
  tests/testbatchawaredispatch.cpp
  tests/testnnevalcanary.cpp
  distributed/client.cpp
  command/commandline.cpp
  command/analysis.cpp
  command/benchmark.cpp
  command/benchmarknn.cpp
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
  vcfsolver/VCFsolver.cpp
  vcfsolver/VCFHashTable.cpp
  main.cpp
  )

if(USE_BACKEND STREQUAL "CUDA")
  target_compile_definitions(katago PRIVATE USE_CUDA_BACKEND)
  target_compile_definitions(katago PRIVATE CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
  find_package(CUDAToolkit REQUIRED)
  # Standalone ABI/dispatch contract: intentionally links no generated
  # package. The test supplies a synthetic registry and exercises B24/B28,
  # packed/planar and current-device proof ownership.
  add_executable(c384_h12_fa4_registry_contract EXCLUDE_FROM_ALL
    tests/c384_h12_fa4_registry_contract.cpp
    neuralnet/c384_h12_fa4_sm120.cpp)
  target_include_directories(c384_h12_fa4_registry_contract PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(c384_h12_fa4_registry_contract PRIVATE CUDA::cudart)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
  if((NOT CUDNN_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  if(KATAGO_ENABLE_CUDNN_FRONTEND_SDPA)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/cudnn-frontend/include/cudnn_frontend.h")
      message(FATAL_ERROR
        "KATAGO_ENABLE_CUDNN_FRONTEND_SDPA requires external/cudnn-frontend/include/cudnn_frontend.h")
    endif()
    target_compile_definitions(katago PRIVATE KATAGO_ENABLE_CUDNN_FRONTEND_SDPA=1)
    target_include_directories(katago SYSTEM PRIVATE external/cudnn-frontend/include)
    # cuDNN Frontend's experimental OSS-engine headers contain inline NVRTC/driver references.
    # They are pulled into this translation unit even though SDPA uses cuDNN heuristics.
    target_link_libraries(katago CUDA::cublas CUDA::cublasLt ${CUDNN_LIBRARY} CUDA::nvrtc CUDA::cuda_driver)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set_source_files_properties(neuralnet/cudabackend.cpp PROPERTIES COMPILE_OPTIONS "-Wno-null-dereference")
    endif()
  else()
    target_link_libraries(katago CUDA::cublas ${CUDNN_LIBRARY})
  endif()
  if(KATAGO_ENABLE_SM120_TRANSFORMER_WINNER)
    target_compile_definitions(katago PRIVATE
      KATAGO_ENABLE_RENJU15_RMS_SM120=1
      KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120=1
      KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120=1
      KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120=1
      KATAGO_ENABLE_RENJU15_FA4_SM120=1
      KATAGO_ENABLE_C384_EXACT_FIXED_AOT=1
      "KATAGO_C384_EXACT_QKV_TACTIC_ID=\"${KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID}\""
      "KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID=\"${KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID}\"")
    if(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT)
      target_compile_definitions(katago PRIVATE
        KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT=1)
    endif()
    target_include_directories(katago PRIVATE
      "${KATAGO_RENJU15_FA4_B36_AOT_ROOT}")
    if(KATAGO_C384_H12_FA4_INCLUDE_DIRS)
      target_include_directories(katago PRIVATE
        ${KATAGO_C384_H12_FA4_INCLUDE_DIRS})
    endif()
    target_link_libraries(katago
      "${KATAGO_RENJU15_FA4_B36_OBJECT}"
      "${KATAGO_RENJU15_FA4_CUTE_RUNTIME}")
    if(KATAGO_C384_H12_FA4_ENABLED)
      target_link_libraries(katago
        ${KATAGO_C384_H12_FA4_OBJECTS}
        "${KATAGO_C384_H12_FA4_RUNTIME}")
    endif()
    message(STATUS
      "C384 exact selection qkv='${KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID}' dual='${KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID}'")
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

