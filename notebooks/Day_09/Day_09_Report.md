# Day 9 Report — Model Optimization & Hyperparameter Tuning

## 🔧 Technical Summary

Today’s focus shifted from simply building models to refining them — making them more reliable and better tuned.

I worked on **hyperparameter tuning** using Ridge Regression. Unlike previous days where the model was trained with default settings, today’s task was about finding the most suitable value for the **alpha parameter**, which controls regularization.

To avoid manual trial and error, I used **GridSearchCV**, which automates the process of testing multiple values and selecting the best one based on performance.

---

## ⚙️ What I Implemented

- Loaded the California Housing dataset  
- Performed train-test split  
- Applied feature scaling using StandardScaler  
- Trained a **default Ridge Regression model**  
- Used **GridSearchCV with 5-fold cross-validation** to search for the best alpha  
- Compared performance using the **R² score**

---

## 📊 Results

- **Best Alpha Found:** 0.1  
- **Best Cross-Validation R² Score:** ~0.611  

### Model Comparison:

| Model              | R² Score |
|------------------|----------|
| Default Ridge     | 0.5758   |
| Optimized Ridge   | 0.5757   |

---

## 🧠 Observations

Interestingly, the optimized model did not outperform the default model significantly. The difference in performance was extremely small.

This suggests that:
- The default model was already close to optimal  
- The dataset does not require heavy regularization tuning  
- The model is stable and not overly sensitive to changes in alpha  

Rather than indicating failure, this actually confirms that the model is **consistent and not overfitting**, which is a good sign in real-world scenarios.

---

## 🔍 Understanding Cross-Validation

Using **5-fold cross-validation** ensured that the model was evaluated fairly across different subsets of the data.

This helps avoid "lucky splits" and gives a more reliable estimate of model performance.

---

## ✍️ Reflection

Starting with a wider range of values helps in exploring different scales of the parameter instead of getting stuck in a narrow region.

If we begin with small increments, we might completely miss better-performing values that lie far from our initial guess. A wider range helps identify the correct region first, and then we can fine-tune within that region for better precision.

---

## 🚀 Key Takeaway

Today was less about improving accuracy and more about understanding **how to systematically optimize a model**.

It reinforced the idea that:
> A good engineer doesn’t just build models — they validate, tune, and understand their behavior.

---