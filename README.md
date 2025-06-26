# 📚 Book Keyword Prediction

A Transformer-based model for predicting future trends in book-related keywords, using time-series analysis on keyword embedding vectors.

---

## 💡 Project Overview

This project focuses on forecasting the future demand for books by analyzing the temporal trends of keywords extracted from book titles. It uses a custom sliding window method on historical keyword vectors, and applies a Transformer model to generate the next keyword sequence. This can help libraries or publishers make informed purchasing decisions.

---

## 🔧 Features

- 🔠 **Keyword Extraction & Cleaning** from book title data  
- 🪄 **Sliding Window Data Generation** for time-series modeling  
- 🧠 **Transformer Model** to predict future keyword embeddings  
- 📊 **Top Keyword Mapping** via nearest vector distance  
- 🏷️ **Publisher Analysis** based on keyword frequency  
- 📈 **Visualization Tools** for trend analysis  

---

## 🧪 Sample Workflow

```bash
# Step 1: Train the model
python train.py  # Calls train_origin() or train_slide()

# Step 2: Predict top keywords
python predict.py  # Calls get_predict_array()

