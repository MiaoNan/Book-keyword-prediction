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

## 📁 Project Structure

```
.
├── main.py                     # Model training and prediction
├── functions.py               # Data preprocessing and analysis
├── config.py                  # Configuration paths
├── book_title_vector.csv      # Preprocessed keyword vectors
├── predict_result.csv         # Output predicted keywords
└── test2.jpg                  # Visualization of keyword-publisher distribution
```

---

## 🧪 Sample Workflow

```bash
# Step 1: Train the model
python main.py  # Calls train_origin() or train_slide()

# Step 2: Predict top keywords
python main.py  # Calls get_predict_array()
```

---

## 📊 Output Files

- `predict_result.csv` — Top predicted keywords  
- `word_frequency_count_press_50.csv` — Keyword-to-publisher frequency matrix  
- `test2.jpg` — Bar chart of publisher distribution based on keywords  

---

## 📦 Dependencies

Make sure the following Python packages are installed:

```bash
pip install pandas numpy matplotlib jieba torch
```

---

## 📌 Use Cases

- Forecast popular themes to assist **library procurement planning**
- Support **publisher strategy** by analyzing keyword distribution
- Explore **keyword evolution** using embedding dynamics

---

## 🤝 Contributing

Contributions, suggestions, and ideas are welcome!

---

## 🧑‍💻 Author

MiaoNan / Michael He  
University of Victoria — Master of Data Science