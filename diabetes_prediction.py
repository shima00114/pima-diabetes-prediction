# ==========================================
# Pima Indians Diabetes Prediction
# 糖尿病発症予測モデル構築スクリプト
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report

# ------------------------------------------
# 1. データの読み込みと探索的データ解析 (EDA)
# ------------------------------------------
print("=== 1. Data Loading & EDA ===")

# データセットの読み込み（ファイルパスは環境に合わせて変更してください）
df = pd.read_csv('diabetes.csv')

# 基本情報の確認
print("\n[Data Shape]:", df.shape)
print("\n[First 5 Rows]:")
print(df.head())

print("\n[Data Information]:")
df.info()

print("\n[Basic Statistics]:")
print(df.describe().T)

print("\n[Missing Values]:")
print(df.isnull().sum())

print("\n[Duplicates]:", df.duplicated().sum())

# 目的変数の分布確認
print("\n[Outcome Distribution (0: Non-Diabetic, 1: Diabetic)]:")
print(df['Outcome'].value_counts(normalize=True))

# 相関行列の計算と可視化
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png') # 画像として保存
plt.show()


# ------------------------------------------
# 2. データ前処理 (Preprocessing)
# ------------------------------------------
print("\n=== 2. Data Preprocessing ===")

# 特徴量 (X) と目的変数 (y) の分割
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 学習用データとテスト用データに分割 (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特徴量の標準化（スケーリング）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train dataset size: {X_train.shape[0]} samples")
print(f"Test dataset size: {X_test.shape[0]} samples")


# ------------------------------------------
# 3. モデルの構築・学習・評価
# ------------------------------------------
print("\n=== 3. Model Training & Evaluation ===")

# 比較する機械学習モデルの定義
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    # スケーリング後のデータでモデルの学習
    model.fit(X_train_scaled, y_train)
    
    # 予測
    y_pred = model.predict(X_test_scaled)
    
    # 評価指標の算出
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 評価結果のまとめ
results_df = pd.DataFrame(results)
print("\n=== Model Comparison Results ===")
print(results_df.to_string(index=False))


# ------------------------------------------
# 4. 特徴量重要度の可視化 (Random Forest)
# ------------------------------------------
rf_model = models['Random Forest']
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importances.png') # 画像として保存
plt.show()
