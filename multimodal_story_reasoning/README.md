# Multimodal Story Reasoning

Deep Neural Networks and Learning Systems  
Module Code: 55-710365

---

# Project Overview

This project implements a multimodal deep learning architecture for visual story reasoning using the StoryReasoning dataset.

The system learns to understand sequential image-text stories and predicts coherent multimodal continuations. The architecture combines computer vision, natural language processing, temporal sequence modelling, multimodal fusion, and explainability techniques.

The project was implemented in Google Colab using PyTorch and Hugging Face libraries.

---

# Research Objective

The main objective of this project is to develop a multimodal neural architecture capable of:

- Understanding visual story sequences
- Encoding images and text jointly
- Learning temporal story evolution
- Generating coherent story continuations
- Improving multimodal alignment using contrastive learning
- Providing explainability through GradCAM attention visualization

---

# Dataset

Dataset Used:

StoryReasoning Dataset

Reference:

Oliveira, D. A. P., & Matos, D. M. (2025). StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation.

Dataset Source:

https://huggingface.co/datasets/daniel3303/StoryReasoning

---

# Architecture

The proposed architecture contains the following components:

## 1. Visual Encoder

- CNN-based image encoder
- Extracts visual feature embeddings
- Uses transfer learning from pretrained convolutional layers

## 2. Text Encoder

- DistilBERT transformer encoder
- Generates contextual text embeddings
- Handles sequential story descriptions

## 3. Multimodal Fusion

- Concatenates image and text embeddings
- Fully connected fusion layer
- Learns shared multimodal representation

## 4. Temporal Sequence Modelling

- LSTM-based temporal reasoning
- Captures story evolution across sequences

## 5. Text Decoder

- GRU-based text generation decoder
- Predicts future story continuation

## 6. Image Decoder

- Convolutional transpose decoder
- Generates visual continuation embeddings

---

# Innovation Implemented

The project introduces multiple architectural improvements:

## Contrastive Multimodal Alignment

A contrastive learning objective was added to align visual and textual embeddings inside a shared latent space.

Benefits:
- Improved multimodal consistency
- Better semantic alignment
- Stronger representation learning

## Explainability using GradCAM

GradCAM explainability was implemented to visualize regions of the image influencing model decisions.

Generated outputs:
- Attention heatmaps
- GradCAM overlays
- Explainability visualizations

---

# Training Configuration

| Parameter | Value |
|---|---|
| Embedding Dimension | 256 |
| Hidden Dimension | 512 |
| Batch Size | 4 |
| Epochs | 2 |
| Learning Rate | 0.0001 |
| Optimizer | AdamW |

---

# Evaluation Metrics

The model was evaluated using:

- BLEU Score
- ROUGE-L Score
- METEOR Score

---

# Final Results

| Metric | Score |
|---|---|
| BLEU | 0.5287 |
| ROUGE-L | 0.6729 |
| METEOR | 0.6646 |

The results demonstrate strong multimodal reasoning capability and coherent sequence generation performance.

---

# Explainability Results

The explainability module successfully generated:

- GradCAM overlays
- Attention heatmaps
- Visual reasoning maps

These outputs improve interpretability and demonstrate the areas influencing model predictions.

---

# Repository Structure

```text
multimodal_story_reasoning/

│
├── README.md
├── requirements.txt
├── experiment_notebook.ipynb
├── config.yaml
│
├── checkpoints/
│
├── results/
│   ├── plots/
│   ├── tables/
│   ├── predictions/
│   ├── attention_maps/
│   ├── figures/
│   ├── metrics/
│   ├── gradcam/
│   └── model_outputs/
│
└── src/
    ├── model.py
    ├── train.py
    └── utils.py