---
title: Training pipelines for LLMs vs. embedding models
subtitle: "test test test" 
description: "test test test"
date: 2025-07-03
toc: true
---

LLM Training Pipeline

1. Pretraining (predict next token)
    Type of data
    Goal
2. Instruction tuning
    Type of data
    Goal
3. RLHF
    Type of data
    Goal

Embedding model training pipeline

1. Pretraining (predict masked tokens with MLM)
    Type of data
    type of learning (unsupersived)
    Goal: language understanding
    model that understands language but isnt optimized for any specific task

    input: "The cat [MASK] on the mat"
    target: "The cat sat on the mat"

2. Constrastive fine-tuning
    Type of data: sentence pairs

    Goal: learn semantic similarity

    positive_pairs = [
    ("What is Python?", "Python is a programming language"),
    ("Movie review", "This film was excellent")
]
3. RLHF
    Type of data
    Goal: Optimize for specific task (would this be )

# BERT -> Sentence BERT
1. PRetrainign: BERT with MLM
2. Siamese network trainign with sentence paris

## RoBERTa -> BGE Models
1. Pretraining: RoBERTa with improved MLM
2. large scale contrastive trainignw tih hard negative minimum

# Nomic BERT -> Nomic Embed

1. Pre-train Nomic BERT-2048 with MLM
2. multi-stage contrastive learning on text pairs from weakly related source
