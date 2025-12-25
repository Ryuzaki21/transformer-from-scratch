# Transformer from Scratch (PyTorch)

This repository contains a **from-scratch implementation of the Transformer model**
based on the paper **“Attention Is All You Need”**.

The model is trained on the **English → Hindi** translation task using the
**OPUS-100 (en-hi)** dataset.

---

## Model Implementation Notes

- `model.py` is **written in a simple and explicit manner**
- Code prioritizes **readability and clarity over optimization**
- Variable names and logic are kept easy to follow
- Designed mainly for **study, learning, and understanding Transformers**

This makes the project suitable for:
- Students learning Transformers
- Debugging and experimentation
- Interview preparation

---

## Dataset

- **OPUS-100** English–Hindi (`Helsinki-NLP/opus-100`)
- Tokenizers are trained using **BPE**
- Special tokens used:
  - `[PAD]` – padding
  - `[SOS]` – start of sequence
  - `[EOS]` – end of sequence
  - `[UNK]` – unknown token

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
