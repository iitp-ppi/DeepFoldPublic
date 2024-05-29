# Copyright 2024 DeepFold Team


"""General-purpose errors used throughout the data pipeline"""


class Error(Exception):
    """Base class for exceptions."""


class PDBxError(Error):
    """PDBx parsing errors."""


class PDBxWarning(Warning):
    """PDBx parsing warnings."""


class PDBxConstructionError(PDBxError):
    pass


class PDBxConstructionWarning(PDBxWarning):
    pass
