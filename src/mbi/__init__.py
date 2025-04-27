"""Main entry point for the mbi package.

This module exposes the core classes and submodules of the mbi library,
making them available for direct import. It simplifies access to functionalities
like data representation (Domain, Dataset), factor manipulation (Factor),
and various estimation and oracle modules.
"""
from .domain import Domain
from .dataset import Dataset
from .factor import Factor
from .clique_vector import CliqueVector
from .marginal_loss import LinearMeasurement
from . import estimation
from . import callbacks
from . import junction_tree
from . import marginal_oracles

__all__ = [
    'Domain',
    'Dataset',
    'Factor',
    'CliqueVector',
    'LinearMeasurement',
    'estimation',
    'callbacks',
    'junction_tree',
    'marginal_oracles',
]
