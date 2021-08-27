
import torch
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

compile_args = {"cxx": [], "nvcc": [] }

if torch.__version__ < "1.8":
    version = torch.__version__.split('.')
    compile_args['cxx'] += ["-DTORCH_VERSION_MAJOR={}".format(version[0])]
    compile_args['cxx'] += ["-DTORCH_VERSION_MINOR={}".format(version[1])]

setup(
        name = 'memory_saving',
        version = '1.0', # sync with memory_saving/__init__.py
        author = 'Peng Chen',
        author_email = 'blueardour@gmail.com',
        url = "https://github.com/blueardour/memory_saving",
        packages=find_packages(),
        ext_modules=[
            cpp_extension.CppExtension(
            'memory_saving.native',
            ['native.cpp'],
            extra_compile_args=compile_args,
            ),
            # cpp_extension.CUDAExtension(
            #     'memory_saving.cpp_extension.quantization',
            #     ['memory_saving/cpp_extension/quantization.cc',
            #      'memory_saving/cpp_extension/quantization_cuda_kernel.cu']
            # ),
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension})

