# pima-diabetes-prediction
# 🩺 Pima Indians Diabetes Prediction

ピマ族（Pima Indians）の臨床データを活用し、患者の生理学的特徴から**糖尿病の発症（Outcome: 0または1）を予測する機械学習分類モデル**を作成したプロジェクトです。

---

## 📌 プロジェクト概要
* **目的**: 患者の医療測定値に基づき、糖尿病の発症リスクを高精度に識別・予測する。
* **データセット**: Pima Indians Diabetes Database (Kaggle / 768件, 9変数)
* **手法**: 探索的データ解析 (EDA)、標準化前処理、複数分類アルゴリズムの比較評価

---

## 📊 使用した特徴量 (Features)

| 変数名 | 概要 |
| :--- | :--- |
| **Pregnancies** | 妊娠回数 |
| **Glucose** | ブドウ糖耐性試験における血漿グルコース濃度 |
| **BloodPressure** | 拡張期血圧 (mmHg) |
| **SkinThickness** | 三頭筋の皮下脂肪の厚さ (mm) |
| **Insulin** | 2時間セラムインスリン (mu U/ml) |
| **BMI** | 肥満度指数 (kg/m²) |
| **DiabetesPedigreeFunction** | 糖尿病の遺伝的血統スコア |
| **Age** | 年齢 |
| **Outcome** (Target) | 糖尿病の有無 (0: 非発症, 1: 発症) |

---

## 🛠️ 使用技術 / ライブラリ
* **Language**: Python 3.x
* **Data Processing**: Pandas, NumPy
* **Visualization**: Seaborn, Matplotlib
* **Machine Learning**: Scikit-learn (Logistic Regression, SVM, Random Forest)

---

## 📈 分析の流れと主な結果

1. **データ探索 (EDA)**
   * 相関分析により、`Glucose（血糖値）` および `BMI` が糖尿病発症（`Outcome`）と特に高い相関を示すことを確認。
2. **前処理 (Preprocessing)**
   * 学習用・テスト用の分割 (8:2) および `StandardScaler` による特徴量の標準化を実施。
3. **モデル比較 (Model Evaluation)**
   * ロジスティック回帰、サポートベクターマシン (SVM)、ランダムフォレストの性能（Accuracy, Recall, F1-score）を比較。

---

## 🚀 実行方法

1. リポジトリのクローン
   ```bash
   git clone [https://github.com/your-username/pima-diabetes-prediction.git](https://github.com/your-username/pima-diabetes-prediction.git)
   cd pima-diabetes-prediction
