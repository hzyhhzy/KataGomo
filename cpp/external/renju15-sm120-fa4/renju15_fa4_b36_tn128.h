
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} r15fa4b36n128_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_r15fa4b36n128_cuda_init(void **);
void _mlir_r15fa4b36n128_cuda_load_to_device(void **);
static inline void r15fa4b36n128_Kernel_Module_Load(r15fa4b36n128_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_r15fa4b36n128_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_r15fa4b36n128_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void r15fa4b36n128_Kernel_Module_Unload(r15fa4b36n128_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
} r15fa4b36n128_Tensor_mQ_t;


typedef struct {
    void *data;
} r15fa4b36n128_Tensor_mK_t;


typedef struct {
    void *data;
} r15fa4b36n128_Tensor_mV_t;


typedef struct {
    void *data;
} r15fa4b36n128_Tensor_mO_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_r15fa4b36n128__mlir_ciface_cutlass___call___flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__Tensorgmemo3622583257600256321_Tensorgmemo3622583257600256321_Tensorgmemo3622583257600256321_Ten(void **args, int32_t num_args);

static inline int32_t cute_dsl_r15fa4b36n128_wrapper(r15fa4b36n128_Kernel_Module_t *module, r15fa4b36n128_Tensor_mQ_t *mQ, r15fa4b36n128_Tensor_mK_t *mK, r15fa4b36n128_Tensor_mV_t *mV, r15fa4b36n128_Tensor_mO_t *mO, float softmax_scale, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[7] = {
        mQ, mK, mV, mO, &softmax_scale, &stream,
        &ret
    };
    _mlir_r15fa4b36n128__mlir_ciface_cutlass___call___flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__Tensorgmemo3622583257600256321_Tensorgmemo3622583257600256321_Tensorgmemo3622583257600256321_Ten(args, 7);
    return ret;
}
