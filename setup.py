
import torch
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(
        name = 'memory_saving',
        version = '1.0',
        author = 'Peng Chen',
        #author-email = 'blueardour@gmail.com',
        url = "https://github.com/blueardour/memory_saving",
        packages=find_packages(),
        ext_modules=[cpp_extension.CppExtension(
            '_ms.native',
            ['native.cpp'],
            #extra_compile_args=['-DTORCH_VERSION_MAJOR 1 -D TORCH_VERSION_MINOR 8'],
            )],
        cmdclass={'build_ext': cpp_extension.BuildExtension})

