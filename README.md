# Titanic Survival Prediction 🚢

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning techniques. The dataset was cleaned, transformed, and analyzed to build multiple classification models and compare their performance.

---

## Dataset

* Dataset: Titanic Training Dataset (`train.csv`)
* Total Passengers: 891
* Target Variable: `Survived`

  * 0 = Did Not Survive
  * 1 = Survived

---

## Data Preprocessing

The following preprocessing steps were performed:

* Removed `Cabin` column due to excessive missing values.
* Filled missing values in `Age` using the median age.
* Filled missing values in `Embarked` using the most frequent value.
* Extracted passenger titles (Mr, Mrs, Miss, Master, Rare) from names.
* Created `FamilySize` feature.
* Created `IsAlone` feature.
* Created `FarePerPerson` feature.
* Applied Label Encoding to `Sex`.
* Applied One-Hot Encoding to `Embarked` and `Title`.
* Removed unnecessary columns:

  * PassengerId
  * Name
  * Ticket

---

## Feature Engineering

New features created:

* FamilySize
* IsAlone
* Title
* FarePerPerson

These features improved the model's ability to identify survival patterns.

---

## Machine Learning Models Compared

| Model                | Accuracy |
| -------------------- | -------- |
| KNN                  | 82.12%   |
| SVM                  | 81.01%   |
| Gaussian Naive Bayes | 78.21%   |
| Random Forest        | 83.24%   |

---

## Best Model

🏆 **Random Forest Classifier**

Parameters:

* n_estimators = 200
* max_depth = 5
* min_samples_split = 5
* random_state = 42

Accuracy:

* Training Accuracy: 85.81%
* Testing Accuracy: 83.24%

---

## Evaluation Metrics

Random Forest Results:

* Accuracy: 83.24%
* Precision: 82.35%
* Recall: 75.68%
* F1 Score: 78.87%

Confusion Matrix:

```text
[[93 12]
 [18 56]]
```

---

## Feature Importance

Top important features:

1. Fare
2. Age
3. Sex
4. Title_Mr
5. Pclass

These features had the strongest influence on survival prediction.

---

## Model Comparison Chart

Add the generated image:

```markdown
![Model Comparison](screenshots/model_comparison.png)
```

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib

---

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── train.csv
├── titanic.py
├── README.md
└── screenshots/
    └── model_comparison.png
```

---

## Conclusion

Multiple machine learning algorithms were evaluated for Titanic survival prediction. Random Forest achieved the highest accuracy of 83.24% and provided the best overall performance. Feature engineering and proper preprocessing significantly improved prediction quality.

This project demonstrates a complete machine learning workflow including data cleaning, feature engineering, model training, evaluation, and comparison.
