include_guard(GLOBAL)

function(katago_c384_ffn_down_validate_request_policy)
  set(_requested FALSE)
  if(NOT "${KATAGO_C384_EXACT_FFN_DOWN_PACKAGE_CMAKE}" STREQUAL "" OR
     NOT "${KATAGO_C384_EXACT_FFN_DOWN_TACTIC_ID}" STREQUAL "")
    set(_requested TRUE)
  endif()
  if(_requested AND NOT USE_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "C384 exact FFN-down AOT is CUDA-only")
  endif()
  if(NOT "${KATAGO_C384_EXACT_FFN_DOWN_TACTIC_ID}" STREQUAL "" AND
     "${KATAGO_C384_EXACT_FFN_DOWN_PACKAGE_CMAKE}" STREQUAL "")
    message(FATAL_ERROR
      "KATAGO_C384_EXACT_FFN_DOWN_TACTIC_ID requires a generated FFN-down package")
  endif()
endfunction()

function(katago_c384_ffn_down_validate_cuda_version)
  if((NOT "${KATAGO_C384_EXACT_FFN_DOWN_PACKAGE_CMAKE}" STREQUAL "" OR
      NOT "${KATAGO_C384_EXACT_FFN_DOWN_TACTIC_ID}" STREQUAL "") AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR "C384 exact FFN-down AOT requires CUDA 13.0 or newer")
  endif()
endfunction()

function(katago_c384_ffn_down_resolve_manifest_policy)
  foreach(_required IN ITEMS
      KATAGO_C384_FFN_DOWN_PACKAGE_SCHEMA
      KATAGO_C384_FFN_DOWN_PACKAGE_MODE
      KATAGO_C384_FFN_DOWN_SELECTED_BATCH
      KATAGO_C384_FFN_DOWN_SELECTED_ID
      KATAGO_C384_FFN_DOWN_TACTIC_IDS
      KATAGO_C384_FFN_DOWN_REGISTRY_PROVIDER
      KATAGO_C384_FFN_DOWN_GENERATED_INCLUDE_DIR
      KATAGO_C384_FFN_DOWN_GENERATED_HEADERS
      KATAGO_C384_FFN_DOWN_GENERATED_METADATA
      KATAGO_C384_FFN_DOWN_GENERATED_BRIDGES
      KATAGO_C384_FFN_DOWN_GENERATED_OBJECTS
      KATAGO_C384_FFN_DOWN_GENERATED_FILE_SHA256)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
      message(FATAL_ERROR "Generated C384 FFN-down package lacks ${_required}")
    endif()
  endforeach()

  if(NOT KATAGO_C384_FFN_DOWN_PACKAGE_SCHEMA STREQUAL "1")
    message(FATAL_ERROR "C384 FFN-down package schema must be 1")
  endif()
  if(NOT KATAGO_C384_FFN_DOWN_PACKAGE_MODE STREQUAL "PRODUCTION")
    message(FATAL_ERROR "C384 FFN-down package must be PRODUCTION")
  endif()
  if(NOT KATAGO_C384_FFN_DOWN_SELECTED_BATCH STREQUAL "28")
    message(FATAL_ERROR "C384 production FFN-down is fixed to B28/M6300")
  endif()
  set(_production_id
    "c384-s225-residual-ffn-down-m6300-k1024-n384-m128n128k32s3-natural-b28-abi1")
  if(NOT "${KATAGO_C384_FFN_DOWN_SELECTED_ID}" STREQUAL
       "${_production_id}")
    message(FATAL_ERROR
      "C384 FFN-down selected ID is not the promoted typed B28/ABI1 winner")
  endif()
  if(NOT KATAGO_C384_FFN_DOWN_SELECTED_ID MATCHES
       "^[a-z0-9][a-z0-9-]*$")
    message(FATAL_ERROR "Unsafe C384 FFN-down candidate ID")
  endif()

  set(_ids ${KATAGO_C384_FFN_DOWN_TACTIC_IDS})
  list(LENGTH _ids _id_count)
  if(NOT _id_count EQUAL 1)
    message(FATAL_ERROR
      "C384 FFN-down production registry must contain exactly one tactic")
  endif()
  list(GET _ids 0 _only_id)
  if(NOT "${_only_id}" STREQUAL "${KATAGO_C384_FFN_DOWN_SELECTED_ID}")
    message(FATAL_ERROR "C384 FFN-down selected ID is not the sole registry ID")
  endif()
  if(NOT "${KATAGO_C384_EXACT_FFN_DOWN_REQUESTED_ID}" STREQUAL "" AND
     NOT "${KATAGO_C384_EXACT_FFN_DOWN_REQUESTED_ID}" STREQUAL
       "${KATAGO_C384_FFN_DOWN_SELECTED_ID}")
    message(FATAL_ERROR "Requested C384 FFN-down ID disagrees with package")
  endif()
  foreach(_asset_kind IN ITEMS HEADERS METADATA BRIDGES OBJECTS)
    list(LENGTH KATAGO_C384_FFN_DOWN_GENERATED_${_asset_kind} _asset_count)
    if(NOT _asset_count EQUAL 1)
      message(FATAL_ERROR
        "C384 FFN-down production package needs exactly one ${_asset_kind} asset")
    endif()
  endforeach()

  list(GET KATAGO_C384_FFN_DOWN_GENERATED_BRIDGES 0 _bridge)
  get_filename_component(_bridge_dir "${_bridge}" DIRECTORY)
  if(EXISTS "${_bridge_dir}/c384_residual_aot_abi.h")
    message(FATAL_ERROR
      "C384 FFN-down bridge directory shadows the engine-owned residual ABI header")
  endif()

  set(KATAGO_C384_EXACT_EFFECTIVE_FFN_DOWN_TACTIC_ID
    "${KATAGO_C384_FFN_DOWN_SELECTED_ID}" PARENT_SCOPE)
endfunction()

# Search packages remain usable before promotion. As soon as a down package is
# requested, all three production providers must be present and agree on B28.
function(katago_c384_validate_b28_production_bundle_policy)
  if(NOT "${KATAGO_C384_EXACT_FFN_DOWN_PACKAGE_CMAKE}" STREQUAL "")
    if("${KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST}" STREQUAL "" OR
       "${KATAGO_C384_H12_FA4_PACKAGE_CMAKE}" STREQUAL "")
      message(FATAL_ERROR
        "C384 B28 production requires exact QKV/dual, packed FA4, and FFN-down packages together")
    endif()
    if(NOT KATAGO_C384_EXACT_AOT_PACKAGE_MODE STREQUAL "PRODUCTION" OR
       NOT KATAGO_C384_EXACT_AOT_SELECTED_BATCH STREQUAL "28")
      message(FATAL_ERROR "C384 QKV/dual production package must be B28")
    endif()
    if(NOT KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "PRODUCTION" OR
       NOT KATAGO_C384_H12_FA4_BATCHES STREQUAL "28")
      message(FATAL_ERROR "C384 packed FA4 production package must contain only B28")
    endif()
    if(NOT KATAGO_C384_FFN_DOWN_PACKAGE_MODE STREQUAL "PRODUCTION" OR
       NOT KATAGO_C384_FFN_DOWN_SELECTED_BATCH STREQUAL "28")
      message(FATAL_ERROR "C384 FFN-down production package must be B28")
    endif()
  elseif((DEFINED KATAGO_C384_EXACT_AOT_PACKAGE_MODE AND
          KATAGO_C384_EXACT_AOT_PACKAGE_MODE STREQUAL "PRODUCTION") OR
         (DEFINED KATAGO_C384_H12_FA4_PACKAGE_MODE AND
          KATAGO_C384_H12_FA4_PACKAGE_MODE STREQUAL "PRODUCTION"))
    message(FATAL_ERROR
      "Incomplete C384 production package: FFN-down provider is required")
  endif()
endfunction()
