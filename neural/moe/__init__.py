"""Neural generic sparse-MoE runtime layer (SPARSE-2).

Model-agnostic machinery parameterized by ExpertLayout; model adapters live
beside it. Q80 remains the regression anchor: all Q80 modules default to
Q80_LAYOUT and are numerically unchanged.
"""
