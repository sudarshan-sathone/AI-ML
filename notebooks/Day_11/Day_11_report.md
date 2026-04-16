# Day 11 Report — Ensemble Learning (Random Forest)

## 🔧 Technical Summary

Today’s task was focused on moving beyond single models and understanding how ensemble methods can improve both stability and performance.

I implemented a **Random Forest Regressor**, which is essentially a collection of multiple decision trees working together. Instead of relying on a single model, Random Forest combines predictions from many trees to reduce variance and improve generalization.

Using the California Housing dataset, I trained the model and evaluated its performance using the R² score. Compared to simpler models, the Random Forest showed better consistency and predictive capability.

---

## 🌲 Understanding the Model

A single decision tree can easily overfit and is sensitive to small changes in data. Random Forest solves this by:

- Training multiple trees on different subsets of data (Bagging)
- Averaging their predictions
- Reducing overfitting and improving stability

This makes it a strong baseline model for many real-world problems.

---

## 📊 Feature Importance

One of the most useful aspects of Random Forest is its ability to highlight which features matter the most.

I generated a feature importance plot to understand which variables had the highest impact on predictions. This helps in:

- Interpreting the model
- Understanding data better
- Potential feature selection in future tasks

---

## 🧪 Experiment — Tree Count vs Performance

To explore model behavior further, I conducted an experiment by training Random Forest models with different numbers of trees:

- 10 trees  
- 50 trees  
- 200 trees  

For each configuration, I measured:
- R² Score (accuracy)
- Training time

---

## 📈 Observations

- Increasing the number of trees improved performance initially
- The jump from 10 → 50 trees gave noticeable improvement
- Beyond that (50 → 200), improvement was very small
- Training time increased significantly with more trees

---

## 🧠 Key Insight

This experiment clearly demonstrated the concept of **diminishing returns**.

While adding more trees makes the model more robust, after a certain point:
- Accuracy gains become minimal
- Computational cost increases rapidly

This highlights the importance of balancing **performance vs efficiency**, which is a critical engineering consideration.

---

## ✅ Conclusion

Today’s task helped me understand:
- The power of ensemble learning
- Why Random Forest is widely used as a baseline model
- How model complexity affects performance and training time

More importantly, it reinforced the idea that better models are not just about higher accuracy, but also about making efficient and scalable choices.

---