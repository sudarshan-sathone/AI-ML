# Day 5 Report — First Prediction

## 🔧 Technical Summary

Today felt like the first real step into actually *doing* machine learning.

Instead of just preparing data like yesterday, I worked on training a model and making predictions. I started by splitting the dataset into training and testing sets using `train_test_split`. This helped me understand why we don’t test a model on the same data it learns from.

After that, I trained a Linear Regression model. The idea was simple — find the best possible straight line that connects hours studied with exam scores.

Once the model was trained, I used it to make predictions on the test data and compared those predictions with actual values. To check how well the model performed, I calculated the Mean Squared Error (MSE) and R² score.

I got an R² score of **0.97**, which means the model was able to capture the pattern quite well.

Finally, I visualized everything in Jupyter Notebook by plotting the actual data points and the regression line. Seeing that red line fit through the data made things much clearer.

Overall, this felt like moving from “understanding concepts” to actually applying them.

---

## 🐞 Bug Log

There were no major errors today, but I did run into a small warning while predicting new values.

The issue was that the model was trained using a DataFrame, but I initially passed a normal list while predicting. Because of that, Python showed a warning about feature names.

After looking into it, I realized the input format should match the training format. I fixed it by passing the input as a DataFrame with the correct column name.

Once I did that, the warning disappeared.

---

## 💭 Conceptual Reflection

**What happens if the model is trained on only 2 rows instead of 8?**

If the model is trained on only 2 data points, it doesn’t really learn the overall pattern. It just tries to fit those two points, which is not enough to understand the relationship properly.

Because of this, the predictions become unreliable. The model might work for those exact points but will fail when new data is given.

This made it clear that having enough data is important for a model to generalize well.

---

## 🧠 Final Thought

Today made me realize that machine learning is not complicated at its core — it's about finding patterns in data.

Even a simple model like linear regression can give good results if the data is clean and the process is followed correctly.

This was the first time it actually felt like I built something that “learns” instead of just running code.