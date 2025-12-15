#pragma once
#include "namespace_config.h"
#include "c10/cuda/CUDAException.h"  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "pat.h"
#include "pat_fwd_kernel.h"
#include "static_switch.h"
#include <cstdlib>

namespace PAT_NAMESPACE {

// Determine if the architecture supports PAT and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_PAT
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

#define PAT_UNSUPPORTED_ARCH printf("FATAL: PAT requires building with sm version sm80-sm90, but was built for < 8.0!");

#define DEFINE_PAT_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, typename... Args> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const pat_fwd_params params)

DEFINE_PAT_FORWARD_KERNEL(pat_fwd_splitkv_kernel) {
    #if defined(ARCH_SUPPORTS_PAT)
        PAT_NAMESPACE::forward<Kernel_traits>(params);
    #else
        PAT_UNSUPPORTED_ARCH
    #endif
}

template<typename fwd_kernel_traits>
void launch(pat_fwd_params &params, cudaStream_t &stream) {
    dim3 grid(params.CTAs, params.h_k);
    // For performance reasons, we need to use at least twice the memory of Q as the maximum value to apply certain MNW.
    constexpr size_t smem_size = fwd_kernel_traits::SmemSize >= fwd_kernel_traits::kSmemQSize*2 ?
                                fwd_kernel_traits::SmemSize : fwd_kernel_traits::kSmemQSize*2;
    // std::cout << "smem_size : " << smem_size << std::endl;
    auto kernel = &pat_fwd_splitkv_kernel<fwd_kernel_traits>;

    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, fwd_kernel_traits::kNWarps*32, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename elem_type, int Headdim>
void pat_run_mha_fwd_splitkv_dispatch(std::vector<pat_fwd_params> &params) {
    cudaStream_t stream = 0;
    HRatio_SWITCH(params[0].h / params[0].h_k, HRatio, [&]() {
        if (params[0].max_split_per_seq == 1) {
            // In this case, we will not run gather_kernel
            MNW_SWITCH(params[0].tile_q, params[0].tile_kv, params[0].Warps, kBlockM, kBlockN, Warps, [&]() {
                launch<fwd_kernel_traits<elem_type, elem_type, kBlockM, kBlockN, Headdim, Warps, HRatio>>(params[0], stream);
            });
        } else {
            const int num_kernels = params.size();
            const char* disableStreamEnv = std::getenv("DISABLE_STREAM");
            std::vector<cudaStream_t> streams;
            if (disableStreamEnv && std::string(disableStreamEnv) == "1") {
//                std::cout << "Running without streams due to DISABLE_STREAM environment variable." << std::endl;
                for (int i = 0; i < num_kernels; i++) {
                    MNW_SWITCH(params[i].tile_q, params[i].tile_kv, params[i].Warps, kBlockM, kBlockN, Warps, [&]() {
                        launch<fwd_kernel_traits<elem_type, float, kBlockM, kBlockN, Headdim, Warps, HRatio> >(params[i], stream);
                    });
                }
            } else {
                for (int i = 0; i < num_kernels; i++) {
                    cudaStream_t stream;
                    cudaStreamCreate(&stream);
                    streams.push_back(stream);
                }
                for (int i = 0; i < num_kernels; i++) {
                    MNW_SWITCH(params[i].tile_q, params[i].tile_kv, params[i].Warps, kBlockM, kBlockN, Warps, [&]() {
                        launch<fwd_kernel_traits<elem_type, float, kBlockM, kBlockN, Headdim, Warps, HRatio> >(params[i], streams[i]);
                    });
                }
            }

            // run gather_kernel on legacy default stream
            dim3 grid_gather(params[0].b, params[0].h);
//            std::cout << "grid_gather : " << grid_gather << std::endl;
            constexpr int WARPS = Headdim == 64 ? 2 : 4;
            GBLOCKM_SWITCH(params[0].max_split_per_seq, [&]() {
                gather_kernel<elem_type, Headdim, BLOCKM, WARPS><<<grid_gather, WARPS*32, 0>>>(params[0]);
            });
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            for (auto &stream: streams) {
                cudaStreamDestroy(stream);
            }
        }
    });
}

}  // namespace PAT_NAMESPACE
