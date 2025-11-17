Machine Learning — Homework 5 (Attention & Transformer)

This repository contains my solutions for Part B – Coding Assignment of Homework 5.

Q1 — Scaled Dot-Product Attention (NumPy)

Implemented in partB_q1.py

Formula used
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V


Features:

    Computes raw attention scores

    Scales scores by √dₖ

    Applies softmax

    Computes weighted context vector

    Includes a NumPy test example

Q2 — Simple Transformer Encoder (PyTorch)

Implemented in partB_q2.py
Components:

    Multi-head self-attention

    Feed-forward network (Linear → ReLU → Linear)

    Add & Norm (residual + layer normalization)

    Output shape tested on: (32, 10, 128)
    Files in This Repository
    partB_q1.py   → Q1: Scaled Dot-Product Attention (NumPy)
partB_q2.py   → Q2: Transformer Encoder Block (PyTorch)
Notes

This repository is created for academic use as part of the Machine Learning course (Fall 2025).
