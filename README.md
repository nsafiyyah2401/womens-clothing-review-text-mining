# 🛍️ Women’s Clothing Review Text Mining System  
### Text Classification • Clustering • Topic Modelling  
**Group Project – Text Mining (KD34103) | Universiti Malaysia Sabah**

This project develops a full text-mining pipeline using the *Women's E-Commerce Clothing Reviews* dataset from Kaggle. It includes data collection, preprocessing, document representation (TF-IDF & BoW), classification, clustering, and topic modelling to uncover customer insights for e-commerce analytics.

---

## 👤 My Contribution (Data Collection & Classification)
This was a **group project**. My main contributions include:

### **✔️ Data Collection**
- Retrieved and validated the full dataset (*Women’s E-Commerce Clothing Reviews*, Kaggle, 23k rows).  
- Performed exploratory analysis on dataset structure, missing values, class imbalance, review length, vocabulary size, and Zipf’s Law.  
- Documented dataset attributes and statistical properties for system design.

### **✔️ Classification Module**
- Built the complete classification workflow for predicting *Recommended IND* (1 = recommended, 0 = not recommended).  
- Implemented Chi-Square feature selection to extract the **top 1,000 most informative features**.  
- Developed and evaluated **Naive Bayes, LinearSVC, and KNN** using cross-validation and test-set evaluation.  
- Generated performance metrics (Accuracy, Precision, Recall, F1-score) and confusion matrices.

---

## 📊 Project Overview

### **1. Data Collection**
Dataset: *Women’s E-Commerce Clothing Reviews* (Kaggle, 2017).  
- 23,486 rows, 11 attributes  
- Contains review text, ratings, product metadata, and recommendation indicator  
- Notable imbalance: **82% recommended**, 18% not recommended  

Text statistics:
- Avg review length ~60 words  
- 38,473 unique words  
- Text follows Zipf’s Law (common stopwords dominate)

---

## 🧹 2. Preprocessing
- Tokenization  
- Lowercasing  
- Stopword removal (NLTK)  
- Removal of non-alphabetic tokens  
- Removal of rare and meaningless tokens  
- Filter words <3 characters  
- Lexicon-based filtering  
- Creation of cleaned dataset for vectorization  

---

## 🧮 3. Document Representation
### **TF-IDF (7,052 features)**
Used for:
- Classification  
- Clustering  

### **Bag-of-Words (4,803 features)**
Used for:
- Topic Modelling (LDA)  
- Interpretability of co-occurring words  

---

## 🤖 4. Algorithms Used
### **Classification**
Models:
- Naive Bayes  
- LinearSVC  
- K-Nearest Neighbours  

Methods:
- Chi-Square feature selection (top 1,000 features)  
- Train/test split  
- Cross-validation  
- Confusion matrices for error analysis  

### **Clustering (K-Means, K=2 to 9)**
Evaluation metrics:
- Davies-Bouldin Score  
- Purity Score  
- Homogeneity & Completeness  
- V-Measure  
- Adjusted Rand Index  

**Best performance: K = 8**

### **Topic Modelling**
- LDA (8 topics)
- BoW matrix
- Top-word extraction  
- Topic coherence evaluation  

---
## 🧹 2. Preprocessing
- Tokenization  
- Lowercasing  
- Stopword removal (NLTK)  
- Removal of non-alphabetic tokens  
- Removal of rare and meaningless tokens  
- Filter words <3 characters  
- Lexicon-based filtering  
- Creation of cleaned dataset for vectorization  

---

## 🧮 3. Document Representation
### **TF-IDF (7,052 features)**
Used for:
- Classification  
- Clustering  

### **Bag-of-Words (4,803 features)**
Used for:
- Topic Modelling (LDA)  
- Interpretability of co-occurring words  

---

## 🤖 4. Algorithms Used
### **Classification**
Models:
- Naive Bayes  
- LinearSVC  
- K-Nearest Neighbours  

Methods:
- Chi-Square feature selection (top 1,000 features)  
- Train/test split  
- Cross-validation  
- Confusion matrices for error analysis  

### **Clustering (K-Means, K=2 to 9)**
Evaluation metrics:
- Davies-Bouldin Score  
- Purity Score  
- Homogeneity & Completeness  
- V-Measure  
- Adjusted Rand Index  

**Best performance: K = 8**

### **Topic Modelling**
- LDA (8 topics)
- BoW matrix
- Top-word extraction  
- Topic coherence evaluation

---

## 🔗 Dataset Link
To comply with usage rights, raw datasets are **NOT** uploaded.

Kaggle Dataset:  
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews  

---

## 🛠️ Tools & Libraries
- Python  
- pandas, numpy  
- scikit-learn  
- NLTK  
- matplotlib / seaborn  

---

## 📚 Course Information
**KD34103 – Text Mining**  
Universiti Malaysia Sabah (UMS), 2025  

---

## 🙌 Acknowledgement
This system was developed as a group assignment.  
Each member contributed to different modules of the system.  
This repository reflects **my individual contributions**:  
*Data collection • Exploratory analysis • Classification pipeline*



