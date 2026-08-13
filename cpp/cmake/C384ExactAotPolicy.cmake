# Pure-CMake, GPU-free policy checks for the C384 exact-AOT package boundary.
# Keep these checks in a scriptable module so negative configure contracts can
# be exercised with `cmake -P` without enabling CUDA or locating dependencies.

function(katago_c384_exact_validate_request_policy)
  set(_katago_c384_exact_requested FALSE)
  foreach(_katago_c384_exact_input IN ITEMS
      KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST
      KATAGO_C384_EXACT_QKV_TACTIC_ID
      KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID
      KATAGO_C384_H12_FA4_PACKAGE_CMAKE)
    if(NOT "${${_katago_c384_exact_input}}" STREQUAL "")
      set(_katago_c384_exact_requested TRUE)
    endif()
  endforeach()

  if(_katago_c384_exact_requested AND NOT USE_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR
      "C384 exact-AOT/FA4 inputs are valid only with USE_BACKEND=CUDA")
  endif()
  if(_katago_c384_exact_requested AND
     NOT KATAGO_ENABLE_SM120_TRANSFORMER_WINNER)
    message(FATAL_ERROR
      "C384 exact-AOT/FA4 inputs require KATAGO_ENABLE_SM120_TRANSFORMER_WINNER=1")
  endif()
  if(KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST STREQUAL "" AND
     (NOT KATAGO_C384_EXACT_QKV_TACTIC_ID STREQUAL "" OR
      NOT KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID STREQUAL ""))
    message(FATAL_ERROR
      "C384 exact candidate IDs require a generated exact-AOT manifest")
  endif()
  if(NOT KATAGO_C384_EXACT_QKV_TACTIC_ID STREQUAL "" AND
     KATAGO_C384_H12_FA4_PACKAGE_CMAKE STREQUAL "")
    message(FATAL_ERROR
      "An exact packed QKV tactic requires a C384/H12 FA4 package")
  endif()
endfunction()

function(katago_c384_exact_validate_cuda_version)
  set(_katago_c384_exact_requested FALSE)
  foreach(_katago_c384_exact_input IN ITEMS
      KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST
      KATAGO_C384_EXACT_QKV_TACTIC_ID
      KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID
      KATAGO_C384_H12_FA4_PACKAGE_CMAKE)
    if(NOT "${${_katago_c384_exact_input}}" STREQUAL "")
      set(_katago_c384_exact_requested TRUE)
    endif()
  endforeach()
  if(_katago_c384_exact_requested AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR
      "C384 exact-AOT/FA4 inputs require CUDA 13.0 or newer")
  endif()
endfunction()

function(katago_c384_exact_resolve_manifest_policy)
  foreach(_katago_c384_exact_forbidden IN ITEMS
      KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID
      KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID)
    if(DEFINED ${_katago_c384_exact_forbidden} AND
       NOT "${${_katago_c384_exact_forbidden}}" STREQUAL "")
      message(FATAL_ERROR
        "Generated exact-AOT manifest must not set ${_katago_c384_exact_forbidden}")
    endif()
  endforeach()
  foreach(_katago_c384_exact_required IN ITEMS
      KATAGO_C384_EXACT_AOT_PACKAGE_SCHEMA
      KATAGO_C384_EXACT_AOT_PACKAGE_MODE
      KATAGO_C384_EXACT_AOT_SELECTED_BATCH
      KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID
      KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID
      KATAGO_C384_EXACT_AOT_SELECTED_FAMILIES
      KATAGO_C384_EXACT_AOT_QKV_ROPE_IDS
      KATAGO_C384_EXACT_AOT_DUAL_FFN_IDS)
    if(NOT DEFINED ${_katago_c384_exact_required} OR
       "${${_katago_c384_exact_required}}" STREQUAL "")
      message(FATAL_ERROR
        "Generated exact-AOT manifest lacks ${_katago_c384_exact_required}")
    endif()
  endforeach()

  if(NOT KATAGO_C384_EXACT_AOT_PACKAGE_SCHEMA STREQUAL "1")
    message(FATAL_ERROR "Generated exact-AOT package schema must be 1")
  endif()
  if(NOT KATAGO_C384_EXACT_AOT_PACKAGE_MODE STREQUAL "SEARCH_PAIR" AND
     NOT KATAGO_C384_EXACT_AOT_PACKAGE_MODE STREQUAL "PRODUCTION")
    message(FATAL_ERROR
      "Generated exact-AOT package mode must be SEARCH_PAIR or PRODUCTION")
  endif()
  if(NOT KATAGO_C384_EXACT_AOT_SELECTED_BATCH STREQUAL "24" AND
     NOT KATAGO_C384_EXACT_AOT_SELECTED_BATCH STREQUAL "28")
    message(FATAL_ERROR "Generated exact-AOT package must select B24 or B28")
  endif()
  if(NOT KATAGO_C384_EXACT_AOT_SELECTED_FAMILIES STREQUAL
       "QKV_ROPE;DUAL_FFN")
    message(FATAL_ERROR
      "Generated exact-AOT package must contain QKV_ROPE and DUAL_FFN")
  endif()

  set(_katago_c384_exact_qkv_ids
    ${KATAGO_C384_EXACT_AOT_QKV_ROPE_IDS})
  set(_katago_c384_exact_dual_ids
    ${KATAGO_C384_EXACT_AOT_DUAL_FFN_IDS})
  list(LENGTH _katago_c384_exact_qkv_ids _katago_c384_exact_qkv_count)
  list(LENGTH _katago_c384_exact_dual_ids _katago_c384_exact_dual_count)
  if(NOT _katago_c384_exact_qkv_count EQUAL 1 OR
     NOT _katago_c384_exact_dual_count EQUAL 1)
    message(FATAL_ERROR
      "Generated exact-AOT package must contain exactly one ID per family")
  endif()

  list(FIND _katago_c384_exact_qkv_ids
    "${KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID}"
    _katago_c384_exact_qkv_index)
  list(FIND _katago_c384_exact_dual_ids
    "${KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID}"
    _katago_c384_exact_dual_index)
  if(_katago_c384_exact_qkv_index EQUAL -1)
    message(FATAL_ERROR
      "Selected exact-AOT QKV ID is absent from the QKV_ROPE family")
  endif()
  if(_katago_c384_exact_dual_index EQUAL -1)
    message(FATAL_ERROR
      "Selected exact-AOT dual ID is absent from the DUAL_FFN family")
  endif()

  if(NOT KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID MATCHES
       "-qkv1152-rope192-" OR
     NOT KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID MATCHES
       "-b${KATAGO_C384_EXACT_AOT_SELECTED_BATCH}-abi1$")
    message(FATAL_ERROR
      "Selected exact-AOT QKV ID has the wrong family or fixed batch")
  endif()
  if(NOT KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID MATCHES
       "-f1024-swiglu-" OR
     NOT KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID MATCHES
       "-b${KATAGO_C384_EXACT_AOT_SELECTED_BATCH}-abi1$")
    message(FATAL_ERROR
      "Selected exact-AOT dual ID has the wrong family or fixed batch")
  endif()

  foreach(_katago_c384_exact_selected_id IN ITEMS
      "${KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID}"
      "${KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID}")
    if(NOT _katago_c384_exact_selected_id MATCHES
         "^[a-z0-9][a-z0-9-]*$")
      message(FATAL_ERROR
        "Unsafe generated exact-AOT candidate ID: ${_katago_c384_exact_selected_id}")
    endif()
  endforeach()

  foreach(_katago_c384_exact_asset_list IN ITEMS
      KATAGO_C384_EXACT_AOT_GENERATED_HEADERS
      KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS
      KATAGO_C384_EXACT_AOT_GENERATED_BRIDGES
      KATAGO_C384_EXACT_AOT_GENERATED_METADATA)
    list(LENGTH ${_katago_c384_exact_asset_list}
      _katago_c384_exact_asset_count)
    if(NOT _katago_c384_exact_asset_count EQUAL 2)
      message(FATAL_ERROR
        "Generated exact-AOT package must collapse to two ${_katago_c384_exact_asset_list} assets")
    endif()
  endforeach()

  if(NOT "${KATAGO_C384_EXACT_REQUESTED_QKV_TACTIC_ID}" STREQUAL "" AND
     NOT "${KATAGO_C384_EXACT_REQUESTED_QKV_TACTIC_ID}" STREQUAL
       "${KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID}")
    message(FATAL_ERROR
      "Requested exact-AOT QKV ID disagrees with the generated manifest")
  endif()
  if(NOT "${KATAGO_C384_EXACT_REQUESTED_DUAL_FFN_TACTIC_ID}" STREQUAL "" AND
     NOT "${KATAGO_C384_EXACT_REQUESTED_DUAL_FFN_TACTIC_ID}" STREQUAL
       "${KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID}")
    message(FATAL_ERROR
      "Requested exact-AOT dual ID disagrees with the generated manifest")
  endif()

  if(KATAGO_C384_EXACT_AOT_PACKAGE_MODE STREQUAL "SEARCH_PAIR")
    if(NOT KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "BATCH_SEARCH")
      message(FATAL_ERROR
        "SEARCH_PAIR exact-AOT requires a BATCH_SEARCH FA4 package")
    endif()
    if(NOT "${KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE}" STREQUAL "" OR
       NOT "${KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE_SHA256}" STREQUAL "")
      message(FATAL_ERROR
        "SEARCH_PAIR exact-AOT must not claim production promotion evidence")
    endif()
  else()
    if(NOT KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "PRODUCTION")
      message(FATAL_ERROR
        "PRODUCTION exact-AOT requires a PRODUCTION FA4 package")
    endif()
    if(NOT EXISTS "${KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE}")
      message(FATAL_ERROR
        "PRODUCTION exact-AOT promotion evidence is missing")
    endif()
    file(SHA256 "${KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE}"
      _katago_c384_exact_promotion_sha256)
    if(NOT _katago_c384_exact_promotion_sha256 STREQUAL
         KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE_SHA256)
      message(FATAL_ERROR
        "PRODUCTION exact-AOT promotion evidence SHA256 mismatch")
    endif()
  endif()

  list(FIND KATAGO_C384_H12_FA4_BATCHES
    "${KATAGO_C384_EXACT_AOT_SELECTED_BATCH}"
    _katago_c384_exact_fa4_batch_index)
  if(_katago_c384_exact_fa4_batch_index EQUAL -1)
    message(FATAL_ERROR
      "Selected exact-AOT batch is absent from the FA4 package")
  endif()

  set(KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID
    "${KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID}" PARENT_SCOPE)
  set(KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID
    "${KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID}" PARENT_SCOPE)
endfunction()
