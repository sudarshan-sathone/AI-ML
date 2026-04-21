# Project Teammate Compatibility Prediction

## Problem Statement

In collaborative academic environments, pairing students with incompatible working styles, schedules, or skill sets leads to poor project outcomes. This engine predicts whether two students are **compatible project teammates** based on measurable behavioral and skill-related features — removing guesswork from team formation.

---

## Features Used

| Feature | Range | Description |
|---|---|---|
| `skill_diff` | 0 – 10 | Absolute difference in skill levels between the two students |
| `common_tech` | 0 – 5 | Number of shared technologies both students know |
| `availability_match` | 0 or 1 | Whether their schedules overlap (1 = yes) |
| `communication_score` | 1 – 10 | Self-reported collaboration ability |
| `experience_gap` | 0 – 5 | Difference in years of experience |

**Target:** `compatibility` — `1` if the pair should work together, `0` otherwise.

Dataset: 700 rows, balanced classes (357 not compatible, 343 compatible).

---

## Model Explanation

### Step 1 — Polynomial Features (degree = 2, interaction_only = True)
Raw features are expanded into all pairwise interaction terms (e.g., `availability_match × communication_score`). This captures non-linear relationships between features that a plain linear model would miss entirely.

### Step 2 — Random Forest Classifier
An ensemble of 400 decision trees is trained on the expanded feature set. Configuration:
- `max_depth=10` to prevent overfitting
- `min_samples_split=4`, `min_samples_leaf=2` for stability
- `class_weight="balanced"` to handle any class imbalance

### Step 3 — Train / Test Split (80 / 20)
560 training samples, 140 test samples, stratified to preserve class balance.

---

## Results

```
Accuracy : 88.57%
Recall   : 85.51%
F1 Score : 88.06%

Confusion Matrix:
                  Predicted 0   Predicted 1
Actual 0               65              6
Actual 1               10             59

TN: 65  FP: 6
FN: 10  TP: 59
```

### Classification Report

```
                    precision    recall  f1-score   support
Not Compatible (0)       0.87      0.92      0.89        71
    Compatible (1)       0.91      0.86      0.88        69
          accuracy                           0.89       140
```

### Top Feature Importances

| Rank | Feature | Importance |
|------|---|---|
| 1 | `common_tech × communication_score` | 0.1194 |
| 2 | `availability_match × communication_score` | 0.1024 |
| 3 | `skill_diff × experience_gap` | 0.1007 |
| 4 | `experience_gap` | 0.0886 |
| 5 | `common_tech` | 0.0708 |

**Key Insight:** Interaction terms dominate the top 3 spots — proving that Polynomial Features were essential, not decorative. A student's communication score matters much more *in combination with* their shared tech stack or availability than it does on its own.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the model

```bash
python src/model.py
```

Results are printed to the terminal and saved to `outputs/results.txt`.

---

## Project Structure

```
project-teammate-compatibility/
│
├── data/
│   └── dataset.csv          # Dataset (700 rows)
├── src/
│   └── model.py             # Full ML pipeline
├── outputs/
│   └── results.txt          # Accuracy, Recall, F1, confusion matrix, feature importance
├── README.md
└── requirements.txt
```
