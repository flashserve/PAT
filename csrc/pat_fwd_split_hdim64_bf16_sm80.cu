#include "namespace_config.h"
#include "pat_fwd_launch_template.h"

namespace PAT_NAMESPACE {

template void pat_run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64>(std::vector<pat_fwd_params> &params);

}