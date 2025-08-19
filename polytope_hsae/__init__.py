"""
Polytope Discovery & Hierarchical SAE Integration

A research framework for validating geometric structure of categorical and hierarchical
concepts in LLMs and incorporating that structure into Hierarchical Sparse Autoencoders.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from . import geometry
from . import estimators
from . import validation
from . import concepts
from . import activations
from . import models
from . import training
from . import steering
from . import metrics
# from . import visualization  # Module not yet implemented

__all__ = [
    "geometry",
    "estimators",
    "validation",
    "concepts",
    "activations",
    "models",
    "training",
    "steering",
    "metrics",
    # "visualization",  # Module not yet implemented
]
