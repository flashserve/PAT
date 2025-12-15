import sys
import os.path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

import torch
from torch.utils.cpp_extension import TORCH_LIB_PATH, library_paths

HERE = os.path.abspath(os.path.dirname(__file__))
CSRC_ROOT = os.path.join(HERE, "csrc")
CUTLASS_ROOT = os.environ.get("CUTLASS_ROOT", None)
assert CUTLASS_ROOT is not None, "Please set CUTLASS_ROOT environment variable to the path of CUTLASS repository"

setup(
    name='prefix_attn',
    version='0.1.1',
    packages=['prefix_attn'],
    ext_modules=[
        CUDAExtension(
            name='prefix_attn._prefix_attn',
            sources=[
                'csrc/bindings.cpp',
                'csrc/pat_fwd_split_hdim64_bf16_sm80.cu',
                'csrc/pat_fwd_split_hdim64_fp16_sm80.cu',
                'csrc/pat_fwd_split_hdim128_bf16_sm80.cu',
                'csrc/pat_fwd_split_hdim128_fp16_sm80.cu',
            ],
            include_dirs=[
                CSRC_ROOT,
                os.path.join(CUTLASS_ROOT, 'include'),
                os.path.join(CUTLASS_ROOT, 'tools', 'util', 'include'),
            ],
            library_dirs=[TORCH_LIB_PATH],
            runtime_library_dirs=[TORCH_LIB_PATH],
            extra_link_args=[f'-Wl,-rpath,{path}' for path in library_paths()],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++20'],
                'nvcc': [
                    '-std=c++20',
                    '-arch=compute_80',
                    '-O3',
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-lineinfo",
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.9',
    install_requires=[
        "torch",
        "einops",
        "pandas"
    ]
)