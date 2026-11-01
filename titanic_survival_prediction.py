"""
Titanic Survival Prediction
Author: Sahil Murshid C K
Tools: Python, Pandas, Matplotlib, Seaborn, Scikit-learn

Goal: Predict whether a passenger survived the Titanic disaster
using passenger details such as age, sex, class and fare.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(URL)
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------
print("\nMissing values:\n", df.isnull().sum())

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df = df.drop(columns=["deck", "embark_town", "alive", "class", "who",
                      "adult_male", "alone"])

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.barplot(x="sex", y="survived", data=df, ax=axes[0])
axes[0].set_title("Survival Rate by Gender")
sns.barplot(x="pclass", y="survived", data=df, ax=axes[1])
axes[1].set_title("Survival Rate by Passenger Class")
sns.histplot(data=df, x="age", hue="survived", bins=25, ax=axes[2])
axes[2].set_title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.close()

print("\nSurvival rate by gender:\n", df.groupby("sex")["survived"].mean())
print("\nSurvival rate by class:\n", df.groupby("pclass")["survived"].mean())

# ---------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["family_size"] = df["sibsp"] + df["parch"] + 1
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 5. MODEL TRAINING & COMPARISON
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv = cross_val_score(model, X, y, cv=5).mean()
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"Test accuracy: {acc:.4f} | 5-fold CV accuracy: {cv:.4f}")
    print(classification_report(y_test, preds))

# ---------------------------------------------------------------
# 6. FEATURE IMPORTANCE + CONFUSION MATRIX (best model)
# ---------------------------------------------------------------
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", figsize=(7, 4), title="Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, rf.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nTop features:\n", importance.sort_values(ascending=False).head(5))
print("\nFinal results:", results)
