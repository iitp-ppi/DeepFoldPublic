"""Setup for pip package."""

import importlib.util
import os

import setuptools

spec = importlib.util.spec_from_file_location("package_info", "deepfold/package_info.py")
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

__description__ = package_info.__description__
__license__ = package_info.__license__
__package_name__ = package_info.__package_name__
__version__ = package_info.__version__


def req_file(filename: str, folder: str = "deepfold"):
    """Read requiremnts file."""
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    return [x.strip() for x in content]


# install_requires = req_file("requirements.txt")

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
    packages=["deepfold"],
    # install_requires=install_requires,
    include_package_data=True,
)
