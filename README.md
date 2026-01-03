# QLoRA Fine-Tuning for Price Prediction (LLaMA 3.1)

This project demonstrates **parameter-efficient fine-tuning (QLoRA)** of **LLaMA 3.1 (8B)** for **numeric price prediction** using natural language product descriptions.

The work focuses on:
- Efficient fine-tuning with 4-bit quantization
- Careful evaluation of inference strategies for numeric outputs
- Clean, reproducible experimentation using Hugging Face and TRL

---

## Project Overview

Large Language Models are not naturally optimized for **precise numeric prediction**.  
In this project, LLaMA 3.1 is fine-tuned using **QLoRA**, and different inference strategies are evaluated to improve stability and accuracy when predicting prices.

Key ideas explored:
- QLoRA fine-tuning with minimal GPU memory
- Comparison of **greedy decoding** vs **probability-weighted top-K inference**
- Error analysis using absolute error and RMSLE

---

## Model & Dataset

- **Base Model:** `meta-llama/Meta-Llama-3.1-8B`
- **Fine-Tuning Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Dataset:** `ed-donner/pricer-data`
- **Task:** Predict product price from textual description

---

## Repository Structure
```text
.
├── notebooks/
│   ├── 01_training_qlora.ipynb
│   └── 02_inference_and_evaluation.ipynb
│
├── outputs/
│   ├── greedy_scatter.png
│   └── topk_weighted_scatter.png
│
└── README.md


---

## Notebooks

### 1️ Training – QLoRA Fine-Tuning  
**`training_qlora.ipynb`**

This notebook covers:
- Dataset loading and preprocessing
- 4-bit quantization using `bitsandbytes`
- LoRA configuration for attention projections
- Fine-tuning with `TRL SFTTrainer`
- Logging with Weights & Biases
- Pushing LoRA adapters to Hugging Face Hub

---

### 2️ Inference & Evaluation  
**`inference_and_evaluation.ipynb`**

This notebook evaluates the fine-tuned model using:
- **Greedy decoding** (argmax next token)
- **Probability-weighted top-K inference (K=3)**

Evaluation includes:
- Absolute error
- RMSLE
- Hit-rate based error thresholds
- Scatter plots of predictions vs ground truth

---

## Evaluation Results

Evaluation plots are available in the `outputs/` directory.

- `greedy_scatter.png` shows higher variance in predictions using greedy decoding.
- `topk_weighted_scatter.png` shows tighter alignment with ground truth using weighted top-K inference.

Overall, weighted top-K inference produces more stable numeric predictions compared to greedy decoding.

---

## How to Run

### Requirements
- Python 3.9+
- GPU recommended

### Authentication
Set the following environment variables:

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
