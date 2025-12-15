#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "pat.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace PAT_NAMESPACE {
inline void
set_params_base(base_params &params,
                const int batch_size,
                const int seqlen_q,
                const int num_heads,
                const int num_heads_k,
                const int head_size,
                const int page_block_size,
                const int max_split_per_seq,
                const int max_seqs_in_CTA,
                const int max_blocks_in_CTA,
                const at::Tensor q,
                const at::Tensor k,
                const at::Tensor v,
                at::Tensor& out,
                const at::Tensor num_split_per_seq,
                at::Tensor& softmax_lse,
                const float softmax_scale) {
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(1);
    params.kv_row_stride = k.stride(1);
    params.q_head_stride = q.stride(2);
    params.kv_head_stride = k.stride(2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(1);
    params.o_head_stride = out.stride(2);

    params.q_batch_stride = q.stride(0);
    params.kv_batch_stride = k.stride(0);
    params.o_batch_stride = out.stride(0);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.softmax_lse_batch_stride = softmax_lse.stride(0);

    // Set the dimensions.
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.seqlen_q = seqlen_q;
    params.d = head_size;
    params.page_block_size = page_block_size;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.max_split_per_seq = max_split_per_seq;
    params.max_seqs_in_CTA = max_seqs_in_CTA;
    params.max_blocks_in_CTA = max_blocks_in_CTA;

    params.num_split_per_seq_ptr = num_split_per_seq.data_ptr();
}

inline void
set_params_kernel(pat_fwd_params &params,
                  const at::Tensor query_tables,
                  const at::Tensor block_tables,
                  const at::Tensor num_seqs_per_CTA,
                  const at::Tensor CTA_rank,
                  const at::Tensor kv_in_CTA,
                  const int tile_q,
                  const int tile_kv,
                  const int Warps) {
    params.query_tables_ptr = query_tables.data_ptr();
    params.block_tables_ptr = block_tables.data_ptr();
    params.num_seqs_per_CTA_ptr = num_seqs_per_CTA.data_ptr();
    params.CTA_rank_ptr = CTA_rank.data_ptr();
    params.kv_in_CTA_ptr = kv_in_CTA.data_ptr();
    params.block_tables_row_stride = block_tables.stride(0);
    params.query_tables_row_stride = query_tables.stride(0);
    params.CTAs = query_tables.size(0);
    params.tile_q = tile_q;
    params.tile_kv = tile_kv;
    params.Warps = Warps;
}

inline void pat_run_mha_fwd(std::vector<pat_fwd_params> &params) {
    // all kernels have the same base param, is_bf16 and d is in base_param
    FP16_SWITCH(!params[0].is_bf16, [&]() {
        HEADDIM_SWITCH(params[0].d, [&]() {
            pat_run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params);
        });
    });

}

void
pat_mha_fwd_kvcache(const at::Tensor &q,
                    const at::Tensor &kcache,
                    const at::Tensor &vcache,
                    const at::Tensor &num_split_per_seq,
                    const std::vector<at::Tensor> &query_tables,
                    const std::vector<at::Tensor> &block_tables,
                    const std::vector<at::Tensor> &num_seqs_per_CTAs,
                    const std::vector<at::Tensor> &CTA_ranks,
                    const std::vector<at::Tensor> &kv_in_CTAs,
                    const std::vector<std::vector<int>> MNW,
                    const int max_split_per_seq,
                    const int max_seqs_in_CTA,
                    const int max_blocks_in_CTA,
                    const float softmax_scale,
                    at::Tensor &out,
                    std::optional<at::Tensor> timing_ = std::nullopt
                    ) {
    // check device, dtype, and shape
    at::cuda::CUDAGuard device_guard{q.device()};
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    if (max_split_per_seq > 32) {
        throw std::runtime_error("Currently, only supports up to 32 splits per seq. Excessive splitting may reduce efficiency.");
    }
    if (max_split_per_seq <= 0) {
        throw std::runtime_error("there is no case that a seq has no block");
    }

    const auto sizes = q.sizes(); // [b, 1, n, d]
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int n_heads = sizes[2];
    const int head_size_og = sizes[3];

    // Fixme(Jinjun): support more head size
    if (head_size_og != 64 && head_size_og != 128) {
        throw std::runtime_error("head_size_og must be 64 / 128 currently");
    }

    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int n_heads_kv = kcache.size(2);
    TORCH_CHECK(page_block_size % 16 == 0, "Paged KV cache block size must be divisible by 16");
    TORCH_CHECK(n_heads % n_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");
    const int ratio = n_heads / n_heads_kv;
    TORCH_CHECK(ratio == 1 || ratio == 2 || ratio == 4 || ratio == 8,
                "only support QGA with hq/hkv = 1, 2, 4, 8 now.");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");

    CHECK_SHAPE(q, batch_size, seqlen_q, n_heads, head_size_og);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, n_heads_kv, head_size_og);
    CHECK_SHAPE(vcache, num_blocks, page_block_size, n_heads_kv, head_size_og);

    CHECK_DEVICE(num_split_per_seq);
    TORCH_CHECK(num_split_per_seq.dtype() == torch::kInt32, "num_split_per_seq must have dtype torch.int32");
    TORCH_CHECK(num_split_per_seq.stride(-1) == 1, "num_split_per_seq must have contiguous last dimension");
    CHECK_SHAPE(num_split_per_seq, batch_size);

    for (auto& mnw: MNW) {
        if (mnw[2] == 1 && mnw[1] == 128 && page_block_size < 32) {
            throw std::runtime_error("paged kv block must have block_size >=32 for MNW=(16, 128, 1)");
        }
    }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    auto opts = q.options();
    auto softmax_lse = torch::empty({batch_size, n_heads, seqlen_q}, opts.dtype(at::kFloat));

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);

    base_params base_param;
    set_params_base(base_param,
                    batch_size,
                    seqlen_q,
                    n_heads, n_heads_kv,
                    head_size,
                    page_block_size,
                    max_split_per_seq,
                    max_seqs_in_CTA,
                    max_blocks_in_CTA,
                    q_padded, kcache_padded, vcache_padded, out,
                    num_split_per_seq,
                    softmax_lse,
                    softmax_scale
                    );


    auto option_float = torch::TensorOptions().dtype(torch::kFloat).device(q.device());
    std::optional<at::Tensor> first_part_out, first_part_lse;
    if (max_split_per_seq != 1) {
        first_part_out = at::empty({batch_size, n_heads, max_split_per_seq, head_size}, option_float);
        first_part_lse = at::empty({batch_size, n_heads, max_split_per_seq}, option_float);

        base_param.oaccum_ptr = first_part_out.value().data_ptr();
        base_param.softmax_lseaccum_ptr = first_part_lse.value().data_ptr();
        base_param.oaccum_batch_stride = first_part_out.value().stride(0);
        base_param.oaccum_head_stride = first_part_out.value().stride(1);
        base_param.oaccum_rank_stride = first_part_out.value().stride(2);
        base_param.softmax_lseaccum_batch_stride = first_part_lse.value().stride(0);
        base_param.softmax_lseaccum_head_stride = first_part_lse.value().stride(1);
    } else {
        base_param.oaccum_ptr = out.data_ptr();
        base_param.softmax_lseaccum_ptr = softmax_lse.data_ptr();
        base_param.oaccum_batch_stride = out.stride(0);
        base_param.oaccum_head_stride = out.stride(2);
        base_param.oaccum_rank_stride = 1;
        base_param.softmax_lseaccum_batch_stride = softmax_lse.stride(0);
        base_param.softmax_lseaccum_head_stride = softmax_lse.stride(1);
    }

    const int num_kernels = MNW.size();
    std::vector<pat_fwd_params> params;
    for (int i = 0; i < num_kernels; ++i) {
        params.push_back(pat_fwd_params(base_param));
    }

    int accum_CTA = 0;
    for (int i = 0; i < num_kernels; i++) {
        pat_fwd_params &param = params[i];
        const at::Tensor &query_table = query_tables[i];
        const at::Tensor &block_table = block_tables[i];
        const at::Tensor &num_seqs_per_CTA = num_seqs_per_CTAs[i];
        const at::Tensor &CTA_rank = CTA_ranks[i];
        const at::Tensor &kv_in_CTA = kv_in_CTAs[i];
        const int CTAs = query_table.size(0);

        CHECK_DEVICE(query_table);
        TORCH_CHECK(query_table.dtype() == torch::kInt32, "query_tables must have dtype torch.int32");
        TORCH_CHECK(query_table.stride(-1) == 1, "query_tables must have contiguous last dimension");

        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_tables must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_tables must have contiguous last dimension");

        CHECK_DEVICE(num_seqs_per_CTA);
        TORCH_CHECK(num_seqs_per_CTA.dtype() == torch::kInt32, "num_seqs_per_CTA must have dtype torch.int32");
        TORCH_CHECK(num_seqs_per_CTA.stride(-1) == 1, "num_seqs_per_CTA must have contiguous last dimension");
        CHECK_SHAPE(num_seqs_per_CTA, CTAs);

        CHECK_DEVICE(CTA_rank);
        TORCH_CHECK(CTA_rank.dtype() == torch::kInt32, "CTA_rank must have dtype torch.int32");
        TORCH_CHECK(CTA_rank.stride(-1) == 1, "CTA_rank must have contiguous last dimension");
        CHECK_SHAPE(CTA_rank, CTAs);

        CHECK_DEVICE(kv_in_CTA);
        TORCH_CHECK(kv_in_CTA.dtype() == torch::kInt32, "kv_in_CTA must have dtype torch.int32");
        TORCH_CHECK(kv_in_CTA.stride(-1) == 1, "kv_in_CTA must have contiguous last dimension");
        CHECK_SHAPE(kv_in_CTA, CTAs);

        set_params_kernel(param,
                          query_table,
                          block_table,
                          num_seqs_per_CTA,
                          CTA_rank,
                          kv_in_CTA,
                          MNW[i][0],
                          MNW[i][1],
                          MNW[i][2]);
        if (timing_.has_value()) {
            param.timing_ptr = static_cast<char*>(timing_.value().data_ptr()) + accum_CTA * n_heads_kv * 512 * sizeof(uint64_t);
        }
        accum_CTA += CTAs;
    }

    pat_run_mha_fwd(params);
}
}