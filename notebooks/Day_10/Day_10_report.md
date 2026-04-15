# Day 10 Report — Non-Linear Models & Overfitting

## 🔧 Technical Summary

Today’s task focused on understanding how machine learning models handle **non-linear relationships**.

Instead of relying on straight-line assumptions, I explored how models can capture more complex, curved patterns in data. The main comparison was between:

- Polynomial Regression (smooth, continuous curve)
- Decision Tree Regressor (step-like predictions)

---

## 📊 What I Implemented

### 1. Polynomial Regression

I used `PolynomialFeatures` to transform the input data and allow a linear model to learn a curved relationship.

- Generated synthetic non-linear data
- Transformed features to include squared terms
- Trained a linear regression model on transformed data
- Observed a smooth curve that captured the overall trend well

---

### 2. Decision Tree Regressor

I implemented a decision tree model which works differently from regression.

- Instead of fitting a formula, it splits data using conditions
- The output looked like a step function rather than a smooth curve

This helped me understand how trees approximate patterns differently.

---

### 3. Overfitting Experiment (max_depth Analysis)

To study model complexity, I trained decision trees with different depths:

- **max_depth = 2** → Underfitting (too simple)
- **max_depth = 5** → Balanced model
- **max_depth = 20** → Overfitting (very irregular)

The deeper tree tried to fit every single data point, resulting in a highly "jittery" curve.

---

### 4. Model Comparison

I compared both approaches visually and numerically using R² score:

- Polynomial Regression → ~0.85
- Decision Tree (depth=2) → ~0.81
- Decision Tree (depth=5) → ~0.93
- Decision Tree (depth=20) → 1.0

Although the depth=20 tree had a perfect score, it was clearly overfitting.

---

## 🧠 Key Observations

- Polynomial regression provides a **smooth approximation** of the data
- Decision trees create **step-like predictions**
- Increasing tree depth increases model complexity
- Very deep trees can **memorize data instead of learning patterns**

---

## 💭 Reflection

While experimenting with different models, I noticed that the decision tree with a very high depth produced a highly irregular, “jittery” curve that tried to pass through almost every data point. Even though this resulted in a perfect R² score on training data, it made it clear that the model was overfitting.

On the other hand, the polynomial regression produced a smoother curve that didn’t perfectly match every point but captured the overall trend effectively.

This showed me that a model that generalizes well is more useful than one that simply memorizes the training data. A slightly imperfect but stable model is often better for real-world predictions.

---

## ✅ Conclusion

This task helped me understand:

- How to handle non-linear data
- The difference between parametric (regression) and non-parametric (tree) models
- The impact of model complexity
- Why overfitting is harmful despite high accuracy

Overall, it was a good exercise in balancing model performance and generalization.