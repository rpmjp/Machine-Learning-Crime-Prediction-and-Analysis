# 🔍 Crime Pattern Prediction and Patrol Optimization with Machine Learning

## 🧠 Overview

This graduate-level machine learning project analyzes crime data from Washington, D.C., with the goal of developing a predictive model that assists law enforcement agencies in understanding crime patterns and optimizing patrol deployment. By leveraging real-world data and advanced ML techniques, the system aims to serve as a **decision-support tool** for crime prevention and strategic intervention.

---

## 🎯 Objective

To build a reliable and interpretable machine learning pipeline capable of classifying crime types (e.g., violent vs. property offenses) and identifying patterns across time, location, and method. The ultimate goal is to support **proactive policing** by predicting when and where certain types of crimes are likely to occur.

---

## 📊 Dataset

- **Source:** [District of Columbia Metropolitan Police Department (MPD)](https://opendata.dc.gov/)
- **Merged Datasets:** `crimeDC_24.csv` and `crime_dc.csv`
- **Records:** ~290,000 incidents
- **Key Fields:**
  - `OFFENSE_GROUP`: Target variable (e.g., violent, property, other)
  - `METHOD`, `SHIFT`, `PSA`, `HOUR`, `DAY_OF_WEEK`: Key predictors
  - `LATITUDE`, `LONGITUDE`: Spatial data (for future expansion into geospatial clustering)

---

## 🧼 Step 1: Data Cleaning & Preprocessing

- Merged datasets using `ccn` identifier
- Removed duplicates and irrelevant columns (`OCTO_RECORD_ID`, unused coordinates)
- Extracted features such as `hour`, `day_of_week`, and `is_weekend`
- Applied **one-hot encoding** and **standard scaling**
- Performed **PCA** for dimensionality reduction

---

## 📈 Step 2: Exploratory Data Analysis (EDA)

- Visualized crime frequency by:
  - Hour of day
  - Weekday vs weekend
  - Policing shift
- Identified strong correlations between **offense type** and **temporal/behavioral patterns**
- Observed underperformance of location-only features (e.g., PSA, ward) compared to method/time-based features

---

## 🧠 Step 3: Model Training

Trained and validated multiple classifiers using stratified splits and hyperparameter tuning:

| Model               | Purpose                         | Highlights                                  |
|--------------------|----------------------------------|---------------------------------------------|
| Logistic Regression| Baseline classifier              | Fast and interpretable                      |
| Random Forest      | Tree-based ensemble              | Handles feature interactions well           |
| XGBoost            | Gradient boosting                | Strong performance on imbalanced classes    |
| MLP Classifier     | Deep learning                    | Tuned with ReLU, Softmax, and scaled inputs |
| GradientBoosting   | Lightweight ensemble alternative | Competitive with fewer resources            |

---

## ⚖️ Step 4: Class Imbalance Handling

- Applied **SMOTE** and **undersampling**
- Focused on **binary classification** (e.g., violent vs non-violent)
- Evaluated with **precision, recall, F1-score**, and **ROC-AUC**

---

## 🔍 Step 5: Explainability with SHAP

- Used **SHAP TreeExplainer** to interpret feature contributions
- Found that `method_OTHERS`, `hour`, and `shift` were most influential
- Created global and local explanation plots to ensure transparency

---

## 🛡️ Impact & Use Case

This project was designed to assist **police departments** in:

- Anticipating crime hot zones by time and method
- Strategically scheduling patrols based on learned patterns
- Improving response strategies through **data-informed decision-making**

It can serve as a backend engine for **real-time crime dashboards**, **patrol route optimization**, or **community alerts**, helping agencies shift from reactive to proactive policing.

---

## 🧠 Skills Demonstrated

- ✅ Data wrangling and merging large public datasets
- ✅ Feature engineering & PCA
- ✅ Binary & multiclass classification
- ✅ Model tuning and evaluation
- ✅ SHAP-based explainable AI
- ✅ Real-world application in public safety

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, SHAP
- **Notebook Environment:** Jupyter Notebook (locally + Google Colab for comparison)
- **Hardware:** NVIDIA RTX 4090 with CUDA for accelerated model training

---

## 🔮 Future Enhancements

- 📍 Add **spatial clustering** (e.g., DBSCAN, KMeans) for hotspot detection
- 🕒 Integrate **time-series forecasting** for offense trends
- 🗺️ Deploy a live dashboard with Folium or Streamlit for police use
- 🌍 Incorporate **GIS data** and socio-economic features to deepen context

---

## 📝 License

```text
Copyright (c) 2025 Robert Jean Pierre  
All rights reserved.  

This work is protected under copyright law.  
No part of this repository may be used, copied, modified, or distributed without explicit written permission from the author.
