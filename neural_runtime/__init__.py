"""Model/capability dispatch around Neural's optimized heterogeneous MoE engine.

GPT-OSS uses the production fused-graph engine and its original MXFP4 experts.
Other architecture/weight-format combinations use explicitly labelled adapters;
compatibility does not imply that GPT-OSS-specific kernels can execute them.
"""
