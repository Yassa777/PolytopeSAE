"""
Polytope Discovery & Hierarchical SAE Integration

A research framework for validating geometric structure of categorical and hierarchical
concepts in LLMs and incorporating that structure into Hierarchical Sparse Autoencoders.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from . import estimators, geometry, validation

__all__ = [
    "geometry",
    "estimators",
    "validation",
]
