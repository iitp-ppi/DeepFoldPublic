"""Setup for pip package."""

import importlib.util

import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

spec = importlib.util.spec_from_file_location("package_info", "deepfold/package_info.py")
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

__description__ = package_info.__description__
__license__ = package_info.__license__
__package_name__ = package_info.__package_name__
__version__ = package_info.__version__


ext_modules = []
generator_flags = []

major = torch.cuda.get_device_properties(0).major
minor = torch.cuda.get_device_properties(0).minor
gpu_arch = f"{major}{minor}"

cc_flag = []
compute_capabililties = [gpu_arch]

cc_flag = []
compute_capabililties = [gpu_arch]

for cap in compute_capabililties:
    cc_flag.append(f"-gencode=arch=compute_{cap},code=sm_{cap}")
    cc_flag.append(f"-gencode=arch=compute_{cap},code=compute_{cap}")


sources = [
    "csrc/evoformer_attn/attention.cpp",
    "csrc/evoformer_attn/attention_back.cu",
    "csrc/evoformer_attn/attention_cu.cu",
]


ext_modules.append(
    CUDAExtension(
        name="deepfold_kernel",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O2", "-std=c++17", *generator_flags],
            "nvcc": [
                "-O2",
                "-std=c++17",
                "--use_fast_math",
                f"-DGPU_ARCH={gpu_arch}",
                *generator_flags,
                *cc_flag,
            ],
        },
        include_dirs=[
            "csrc/cutlass/include",
        ],
        optional=True,
    )
)


setuptools.setup(
    name=__package_name__,
    version=__version__,
    description=__description__,
    license=__license__,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=setuptools.find_packages(exclude=["tests"]),
    # install_requires=install_requires,
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
