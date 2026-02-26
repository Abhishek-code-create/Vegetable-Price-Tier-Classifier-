# === Learning & Loss Curves Generator for Kalimati Vegetable Price Dataset ===
# Author: Abhi Khatiwada
# Output: charts/training_accuracy_curve.png, charts/training_loss_curve.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# === 1️⃣ Detect project base directory ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === 2️⃣ Load dataset ===
data_path = os.path.join(BASE_DIR, "data", "clean", "kalimati_model.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print(f"✅ Loaded dataset: {data_path}")
print(f"Records: {df.shape[0]}, Features: {df.shape[1]}")

# === 3️⃣ Split features (X) and target (y) ===
y = df["PriceTier"]

# Keep only numeric columns for training
X = df.drop(columns=["PriceTier"], errors="ignore")
X = X.select_dtypes(include=[np.number])

print(f"Using {X.shape[1]} numeric features for training: {list(X.columns)}")

# Encode labels if categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# === 4️⃣ Define model pipeline ===
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# === 5️⃣ Generate learning-curve data ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=cv, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

# === 6️⃣ Ensure charts directory exists ===
charts_dir = os.path.join(BASE_DIR, "charts")
os.makedirs(charts_dir, exist_ok=True)

# === 7️⃣ Plot Training Accuracy Curve ===
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Accuracy")
plt.plot(train_sizes, test_mean, 'o-', color="green", label="Validation Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")
plt.title("Training and Validation Accuracy Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
acc_path = os.path.join(charts_dir, "training_accuracy_curve.png")
plt.savefig(acc_path, dpi=300)
plt.show()

# === 8️⃣ Plot Training Error ("Loss") Curve ===
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, 1 - train_mean, 'o-', color="red", label="Training Error")
plt.plot(train_sizes, 1 - test_mean, 'o-', color="orange", label="Validation Error")
plt.fill_between(train_sizes, (1 - train_mean) - train_std, (1 - train_mean) + train_std, alpha=0.1, color="red")
plt.fill_between(train_sizes, (1 - test_mean) - test_std, (1 - test_mean) + test_std, alpha=0.1, color="orange")
plt.title("Training and Validation Error Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Error (1 - Accuracy)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
loss_path = os.path.join(charts_dir, "training_loss_curve.png")
plt.savefig(loss_path, dpi=300)
plt.show()

print("\n✅ Curves generated successfully:")
print(f"   • {acc_path}")
print(f"   • {loss_path}")