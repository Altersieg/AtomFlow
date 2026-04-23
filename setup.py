# ============================================================================
# setup.py — AtomFlow CUDA Extension Builder
#
# Automatically invoked by pip/setuptools via pyproject.toml.
# Locates CUDA/C++ source files and compiles them into a Python-callable
# shared library using Torch's cpp_extension.
# ============================================================================

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Hardware architecture flags. For RTX 5060 Ti (Blackwell),
# we target compute capability 120 (sm_120).
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-gencode=arch=compute_120,code=sm_120",
    ],
}

setup(
    # Package metadata is read from pyproject.toml.
    ext_modules=[
        CUDAExtension(
            name="atomflow_cuda",
            sources=[
                # List C++/CUDA files here.
                # "csrc/atomflow_ops.cpp",
                # "csrc/linear_attention_kernel.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
