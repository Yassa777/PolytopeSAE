"""
Polytope Discovery & Hierarchical SAE Integration

A research framework for validating geometric structure of categorical and hierarchical
concepts in LLMs and incorporating that structure into Hierarchical Sparse Autoencoders.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from . import (activations, concepts, estimators, geometry, metrics, models,
               steering, training, validation)

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
