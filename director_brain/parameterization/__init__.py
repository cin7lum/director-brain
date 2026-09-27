"""Director Brain parameterization package.

Deterministic feasibility, candidate generation, and selection/abstention
for film execution parameters. Currently supports J_CUT audio offset only.
"""
from director_brain.parameterization.parameterizer import JCutParameterizer

__all__ = ["JCutParameterizer"]
