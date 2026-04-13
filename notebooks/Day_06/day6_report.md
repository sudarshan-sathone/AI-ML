# Day 6 Report — The Binary Decision

## 🔧 Technical Summary

Today’s task marked a shift from predicting continuous values to making binary decisions. Instead of estimating numbers like in regression, the goal was to classify outcomes into categories — in this case, whether a student would "Pass" or "Fail".

I worked on implementing Logistic Regression, which outputs probabilities instead of raw values. What stood out immediately was how the model uses a sigmoid function to compress predictions between 0 and 1, making it suitable for classification tasks.

The dataset used was simple but effective — hours of sleep and coffee consumption were used as features to predict whether a student passed or failed. After organizing the data into a DataFrame, I split it into training and testing sets to ensure the model could be evaluated on unseen data.

Once the model was trained, I generated predictions and evaluated them using a confusion matrix and classification report. The confusion matrix provided a clear visual of how many predictions were correct versus incorrect, while the classification report gave detailed metrics like precision, recall, and F1-score.

Interestingly, the model achieved perfect accuracy on the test set. However, since the dataset is extremely small, this result is not necessarily reliable and may indicate overfitting rather than true performance.

I also explored feature importance using the model’s coefficients. It was clear that hours of sleep had a positive impact on passing, while higher coffee consumption negatively influenced the outcome. This added a layer of interpretability to the model.

Finally, I tested the model with a custom input (3 hours of sleep and 7 cups of coffee), and the model predicted a failure, which aligns well with intuition.

---

## 🐞 Bug Log

There were no major errors during implementation, but I encountered a minor warning related to feature names while making manual predictions.

Initially, I passed the input as a simple list, which caused a mismatch since the model was trained using a DataFrame with column names. I resolved this by converting the input into a DataFrame with the same feature names, ensuring consistency between training and prediction.

Apart from that, the workflow was smooth.

---

## 💭 Conceptual Reflection

In a cancer detection system, a False Negative is significantly more dangerous than a False Positive.

A False Negative means the model predicts that a patient is healthy when they actually have cancer. This can delay diagnosis and treatment, potentially leading to severe consequences or even loss of life.

On the other hand, a False Positive means predicting cancer when the patient is actually healthy. While this may cause temporary stress and require additional testing, it is still a safer outcome compared to missing a real case.

Therefore, in critical applications like healthcare, minimizing False Negatives is far more important than optimizing overall accuracy.

---

## 🧠 Final Thought

Today’s task made it clear that accuracy alone is not enough when evaluating a model.

The confusion matrix was especially insightful because it broke down the model’s performance into specific types of errors rather than just giving a single number. It showed that understanding *what kind of mistakes* a model makes is just as important as how many mistakes it makes.

This felt like a step closer to real-world machine learning, where the focus is not just on building models, but on interpreting and trusting their decisions.