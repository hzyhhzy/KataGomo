function(cpu_ptq_prepare_mlas_qgemm output_source)
  set(mlas_source_dir
    "${CPU_PTQ_ONNXRUNTIME_SOURCE_ROOT}/onnxruntime/core/mlas/lib")
  set(original_kernel "${mlas_source_dir}/qgemm_kernel_avx2.cpp")
  set(original_driver "${mlas_source_dir}/qgemm.h")
  set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/cpuptq_mlas")
  file(MAKE_DIRECTORY "${generated_dir}")

  # Keep the ORT checkout immutable. Copy the two exact v1.23.0 sources into
  # the build tree and apply the small target-specific changes there. Exact
  # anchor checks make an ORT upgrade fail at configure time instead of
  # silently dropping the tuned panel or corrupting MLAS scratch memory.
  file(READ "${original_kernel}" kernel_text)
  set(kernel_anchor
    "    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 48, 256, 384 };")
  string(FIND "${kernel_text}" "${kernel_anchor}" kernel_anchor_offset)
  if(kernel_anchor_offset EQUAL -1)
    string(FIND "${kernel_text}" "KTCP_MLAS_PACKED_STRIDE_N" kernel_already_patched)
    if(kernel_already_patched EQUAL -1)
      message(FATAL_ERROR
        "CPU-PTQ MLAS patch does not match qgemm_kernel_avx2.cpp; use the pinned ONNX Runtime v1.23.0 source")
    endif()
  else()
    set(kernel_replacement [=[    // Fixed-shape runtimes may tune this cache panel per target CPU. The
    // upstream default remains 256 when no override is supplied.
#ifndef KTCP_MLAS_PACKED_STRIDE_N
#define KTCP_MLAS_PACKED_STRIDE_N 256
#endif
#ifndef KTCP_MLAS_PACKED_STRIDE_M
#define KTCP_MLAS_PACKED_STRIDE_M 48
#endif
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{
        KTCP_MLAS_PACKED_STRIDE_M, KTCP_MLAS_PACKED_STRIDE_N, 384 };]=])
    string(REPLACE "${kernel_anchor}" "${kernel_replacement}" kernel_text "${kernel_text}")
  endif()

  file(READ "${original_driver}" driver_text)
  set(driver_anchor
    "    constexpr size_t bufsize = std::max(packASize + packBSize, packedASize) + rowSumSize + colSumSize + zpbSize;")
  string(FIND "${driver_text}" "${driver_anchor}" driver_anchor_offset)
  if(driver_anchor_offset EQUAL -1)
    string(FIND "${driver_text}" "packedBufSize" driver_already_patched)
    if(driver_already_patched EQUAL -1)
      message(FATAL_ERROR
        "CPU-PTQ MLAS patch does not match qgemm.h; use the pinned ONNX Runtime v1.23.0 source")
    endif()
  else()
    set(driver_replacement [=[    constexpr size_t packedRowSumSize =
        UpAlignSize(PackedStrides.M * sizeof(int32_t));
    constexpr size_t packedColSumSize =
        UpAlignSize(PackedStrides.N * sizeof(int32_t));
    constexpr size_t packedZpbSize =
        UpAlignSize(PackedStrides.N * sizeof(int32_t));

    // Packed-B and unpacked-B operations lay out different-sized scratch
    // regions in the same per-thread buffer. Size both complete layouts so a
    // target-specific packed panel cannot overwrite the thread buffer.
    constexpr size_t unpackedBufSize =
        packASize + packBSize + rowSumSize + colSumSize + zpbSize;
    constexpr size_t packedBufSize =
        packedASize + packedRowSumSize + packedColSumSize + packedZpbSize;
    constexpr size_t bufsize = std::max(unpackedBufSize, packedBufSize);]=])
    string(REPLACE "${driver_anchor}" "${driver_replacement}" driver_text "${driver_text}")
  endif()

  set(generated_kernel "${generated_dir}/qgemm_kernel_avx2.cpp")
  file(WRITE "${generated_kernel}" "${kernel_text}")
  file(WRITE "${generated_dir}/qgemm.h" "${driver_text}")
  set(${output_source} "${generated_kernel}" PARENT_SCOPE)
endfunction()
