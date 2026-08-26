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
#ifndef KTCP_MLAS_PACKED_STRIDE_K
#define KTCP_MLAS_PACKED_STRIDE_K 384
#endif
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{
        KTCP_MLAS_PACKED_STRIDE_M,
        KTCP_MLAS_PACKED_STRIDE_N,
        KTCP_MLAS_PACKED_STRIDE_K };]=])
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

  # The packed-B path dominates this backend. Keep the b11/b16-qualified
  # panel for ordinary projections, but use a 1024-wide panel for b24's
  # fused up/gate projection. PackedStrides.N remains the larger value so the
  # shared scratch buffer is always large enough; only the runtime loop step
  # changes with Shape->N.
  set(dynamic_panel_anchor [=[    PackedColumnSumBuffer += RangeStartN;

    //
    // Step through each slice of matrix B along the K dimension.]=])
  set(dynamic_panel_replacement [=[    PackedColumnSumBuffer += RangeStartN;

    const size_t PanelStrideN =
        Shape->N >= KTCP_MLAS_LARGE_N_THRESHOLD
            ? KTCP_MLAS_LARGE_STRIDE_N
            : KTCP_MLAS_SMALL_STRIDE_N;

    //
    // Step through each slice of matrix B along the K dimension.]=])
  string(FIND "${driver_text}" "${dynamic_panel_anchor}" dynamic_panel_offset)
  if(dynamic_panel_offset EQUAL -1)
    message(FATAL_ERROR "CPU-PTQ MLAS dynamic packed-N panel anchor was not found")
  endif()
  string(REPLACE "${dynamic_panel_anchor}" "${dynamic_panel_replacement}"
    driver_text "${driver_text}")

  set(dynamic_fill_anchor
    "            std::fill_n(ColumnSumBuffer, Strides.N, 0);")
  set(dynamic_fill_replacement
    "            std::fill_n(ColumnSumBuffer, PanelStrideN, 0);")
  string(FIND "${driver_text}" "${dynamic_fill_anchor}" dynamic_fill_offset)
  if(dynamic_fill_offset EQUAL -1)
    message(FATAL_ERROR "CPU-PTQ MLAS dynamic packed-N fill anchor was not found")
  endif()
  string(REPLACE "${dynamic_fill_anchor}" "${dynamic_fill_replacement}"
    driver_text "${driver_text}")

  set(dynamic_count_anchor [=[            CountN = std::min(RangeCountN - n, Strides.N);

            if (k == 0) {]=])
  set(dynamic_count_replacement [=[            CountN = std::min(RangeCountN - n, PanelStrideN);

            if (k == 0) {]=])
  string(FIND "${driver_text}" "${dynamic_count_anchor}" dynamic_count_offset)
  if(dynamic_count_offset EQUAL -1)
    message(FATAL_ERROR "CPU-PTQ MLAS dynamic packed-N count anchor was not found")
  endif()
  string(REPLACE "${dynamic_count_anchor}" "${dynamic_count_replacement}"
    driver_text "${driver_text}")

  string(PREPEND driver_text
    "#define KTCP_MLAS_SMALL_STRIDE_N ${CPU_PTQ_MLAS_PACKED_STRIDE_N}\n"
    "#define KTCP_MLAS_LARGE_STRIDE_N ${CPU_PTQ_MLAS_PACKED_STRIDE_N_LARGE}\n"
    "#define KTCP_MLAS_LARGE_N_THRESHOLD ${CPU_PTQ_MLAS_PACKED_STRIDE_N_LARGE_THRESHOLD}\n")

  set(generated_kernel "${generated_dir}/qgemm_kernel_avx2.cpp")
  file(WRITE "${generated_kernel}" "${kernel_text}")
  file(WRITE "${generated_dir}/qgemm.h" "${driver_text}")
  set(${output_source} "${generated_kernel}" PARENT_SCOPE)
endfunction()

function(cpu_ptq_prepare_mlas_platform output_source)
  set(mlas_source_dir
    "${CPU_PTQ_ONNXRUNTIME_SOURCE_ROOT}/onnxruntime/core/mlas/lib")
  set(original_platform "${mlas_source_dir}/platform.cpp")
  set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/cpuptq_mlas")
  file(MAKE_DIRECTORY "${generated_dir}")

  # ORT normally upgrades the U8S8 kernel to AVX-VNNI after detecting the
  # host CPU. An AVX2 qualification build must retain the AVX2 kernel even
  # when it is benchmarked on a newer machine. Keep the source checkout
  # immutable and patch one pinned dispatch condition in the generated copy.
  file(READ "${original_platform}" platform_text)
  set(platform_anchor
    "if ((Cpuid7_1[0] & 0x10) != 0) {")
  string(FIND "${platform_text}" "${platform_anchor}" platform_anchor_offset)
  if(platform_anchor_offset EQUAL -1)
    string(FIND "${platform_text}" "KTCP_MLAS_ALLOW_AVXVNNI" platform_already_patched)
    if(platform_already_patched EQUAL -1)
      message(FATAL_ERROR
        "CPU-PTQ MLAS AVX2 cap does not match platform.cpp; use the pinned ONNX Runtime v1.23.0 source")
    endif()
  else()
    set(platform_replacement
      "if (KTCP_MLAS_ALLOW_AVXVNNI && (Cpuid7_1[0] & 0x10) != 0) {")
    string(REPLACE "${platform_anchor}" "${platform_replacement}"
      platform_text "${platform_text}")
  endif()

  set(generated_platform "${generated_dir}/platform.cpp")
  file(WRITE "${generated_platform}" "${platform_text}")
  set(${output_source} "${generated_platform}" PARENT_SCOPE)
endfunction()
