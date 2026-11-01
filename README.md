# Titanic Survival Prediction

**Author:** Sahil Murshid C K | [LinkedIn](https://www.linkedin.com/in/sahil-murshid-data) | sahilmurshid6@gmail.com

## Problem
Predict whether a passenger survived the Titanic disaster using details such as age, gender, passenger class, fare and family size.

## Dataset
Titanic passenger data (891 rows, 15 columns), from the Seaborn data repository / Kaggle.

## Tools
Python, Pandas, Matplotlib, Seaborn, Scikit-learn

## Steps
1. **Data cleaning:** filled missing `age` with the median and `embarked` with the mode, dropped `deck` (about 77% missing) and duplicate columns.
2. **EDA:** compared survival by gender, class and age (see `eda_plots.png`).
3. **Feature engineering:** created `family_size` and one-hot encoded categorical columns.
4. **Modeling:** trained Logistic Regression and Random Forest with an 80/20 stratified split and 5-fold cross-validation.
5. **Evaluation:** accuracy, precision/recall, confusion matrix, feature importance.

## Key Findings
- Women survived at **74%** vs **19%** for men.
- 1st class survival was **63%** vs **24%** in 3rd class.
- Most important features: fare, gender, age.

## Results
| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|
| Logistic Regression | 80.4% | 79.1% |
| Random Forest | **81.6%** | **80.8%** |

## How to Run
```bash
pip install pandas scikit-learn seaborn matplotlib
python titanic_survival_prediction.py
```

---

## Resume Entry (copy this)

**Titanic Survival Prediction** | Python, Pandas, Scikit-learn, Seaborn | github.com/sahilmurshid6-ds/titanic-survival-prediction
- Cleaned 891 passenger records by imputing missing values and encoding categorical features; engineered a family-size feature
- Performed EDA revealing that gender and passenger class were the strongest survival factors (74% female vs 19% male survival)
- Built and compared Logistic Regression and Random Forest models; the Random Forest achieved **81.6% test accuracy** (80.8% cross-validated)
