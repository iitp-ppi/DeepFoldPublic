# Copyright 2023 DeepFold Team

"""Package constants."""

MAJOR = 0
MINOR = 1
PATCH = 0
PRE_RELEASE = ""

VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "megafold"
__description__ = "MegaFold - a scalable framework for protein models"
__license__ = "Apache-2.0"
