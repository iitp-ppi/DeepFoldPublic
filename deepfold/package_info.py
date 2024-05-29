# Copyright 2024 DeepFold Team


"""Package constants."""


MAJOR = 2
MINOR = 1
PATCH = 0
PRE_RELEASE = ""

VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "DeepFold"
__description__ = "Protein structure prediction framework."
__license__ = "Apache-2.0"
