
# Vegetable-Price-Tier-Classifier-


---

## 📁 Project Structure

```
Project 
│
├── data/
│   ├── raw/
│   │   └── Kalimati_Tarkari_Dataset.csv        ← Raw dataset (280,862 rows)
│   └── clean/
│       ├── kalimati_clean.csv                  ← Cleaned + feature-engineered (252,148 rows)
│       ├── kalimati_model.csv                  ← Top-30 model dataset (97,820 rows)
│       ├── commodity_stats.csv                 ← Per-commodity statistics
│       └── monthly_avg_prices.csv              ← Monthly price time series
│
├── scripts/
│   ├── 01_preprocess.py                        ← Data cleaning, feature engineering, tier labelling
│   ├── 02_eda.py                               ← EDA visualisations (7 charts)
│   ├── 03_train_evaluate.py                    ← KNN + Decision Tree + NB + RF classification
│   ├── 04_clustering.py                        ← K-Means clustering analysis
│   ├── 05_cli.py                               ← Interactive CLI interface
│   └── 06_run_pipeline.py                      ← ONE COMMAND: runs scripts 01→04
│
├── model/
│   ├── best_model.pkl                          ← Trained KNN model
│   ├── best_scaler.pkl                         ← StandardScaler
│   ├── commodity_encoder.pkl                   ← LabelEncoder (vegetable names)
│   ├── label_encoder.pkl                       ← LabelEncoder (tier labels)
│   ├── last_prices.json                        ← Historical prices for lag features
│   ├── meta.json                               ← Model metadata, features, results
│   ├── metrics.csv                             ← All model performance metrics
│   ├── kmeans_model.pkl                        ← K-Means clustering model
│   └── cluster_assignments.csv                 ← Vegetable cluster memberships
│
├── charts/
│   ├── model_comparison.png                    ← 4-model performance comparison
│   ├── cv_scores.png                           ← Cross-validation F1 scores
│   ├── confusion_matrix_best.png               ← KNN confusion matrix
│   ├── knn_hyperparameter_search.png           ← k vs F1/Accuracy
│   ├── feature_importance.png                  ← Decision Tree feature importance
│   ├── cluster_pca.png                         ← K-Means PCA scatter
│   ├── cluster_elbow.png                       ← Elbow method chart
│   ├── cluster_profiles.png                    ← Cluster seasonal profiles
│   └── eda_*.png                               ← EDA charts (7 figures)
│
├── outputs/
│   └── VegetablePriceClassifier_App.html       ← Web-based GUI
│
├── requirements.txt                            ← Python dependencies
└── README.md                                   ← This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (ONE COMMAND)
```bash
python scripts/06_run_pipeline.py
```
Runs: preprocess → EDA → classification training → clustering. ~2–3 minutes.

### 3. Use the CLI
```bash
# Interactive mode
python scripts/05_cli.py

# Single prediction
python scripts/05_cli.py --vegetable "Cauli Local" --date 2024-06-01 --price 45
```



---



---

## 📦 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | Kalimati Tarkari Dataset |
| **Source** | Kalimati Krishi Produce Market Development Committee |
| **Kaggle** | https://www.kaggle.com/datasets/ramkrijal/kalimati-tarkari-dataset |
| **Raw Records** | 280,862 |
| **Date Range** | June 2013 – September 2023 |
| **Vegetables** | 136 commodities |
| **Target** | PriceTier: Low / Medium / High / Very High |

### Preprocessing Steps
1. Type conversion (remove 'Rs ' prefix from price strings)
2. Date parsing (mixed format handling)
3. Unit standardisation (Kg only — removes Doz, Per Piece)
4. IQR outlier removal (5th–95th percentile per commodity)
5. Feature engineering (temporal, cyclical, lag, rolling features)
6. Per-commodity quartile-based tier labelling

---


