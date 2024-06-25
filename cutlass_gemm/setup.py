from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(
            name='cutlass_gemm',
            sources=['cutlass.cpp', 'cutlass_kernel.cu'],
            extra_compile_args={
                'nvcc': [
                    '-gencode=arch=compute_90a, code=sm_90a',
                    '-lineinfo',
                ]
            },
            include_dirs=[
                '/home/adhoq26/cutlass/include',
                '/home/adhoq26/cutlass/tools/util/include',
            ],
            libraries=['cuda'],
            library_dirs=['/usr/local/cuda-12.4/lib64'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)