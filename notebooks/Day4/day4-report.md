# Day 4 Report — Pre-processing Protocol

## 🔧 Technical Summary

Today was less about “building models” and more about understanding how real-world data behaves.

I worked on **data preprocessing**, which included handling missing values, scaling features, and visualizing relationships between variables.

Using Pandas, I identified missing values and applied imputation techniques like filling numerical data with mean values and default placeholders. This helped in maintaining the dataset instead of losing information.

I also used **MinMaxScaler** from Scikit-learn to normalize features into a common range. This made me understand how models can get biased if features are not scaled properly.

Finally, I explored basic visualization using Seaborn to generate a heatmap and observe how features are correlated.

Overall, this felt like working on the “behind-the-scenes” part of ML — not flashy, but very important.

---

## 🐞 Bug Log

I didn’t face any major errors while implementing today’s tasks, mainly because the required libraries were already installed and the setup from previous days was stable.

However, while working through the scripts, I was initially unsure whether the scaling output was correct since the transformed values looked very different from the original data.

After revisiting the concept, I understood that MinMaxScaler converts all values into a 0–1 range, which is expected behavior.

This helped me realize that not all confusion comes from errors — sometimes it comes from not fully understanding the transformation taking place.
---

## 💭 Conceptual Reflection

**Why is it better to fill missing age with mean rather than 0?**

Filling missing values with 0 can introduce unrealistic data into the dataset. For example, an age of 0 doesn’t make sense in most scenarios and can distort the model’s understanding.

Using the mean, on the other hand, keeps the data within a realistic range and maintains the overall distribution.

This makes the dataset more reliable and helps the model learn more accurate patterns.

---

## 🧠 Final Thought

Today made me realize that most of the actual work in AI is not about building models, but preparing the data properly.

The cleaner the data, the better the model performs.
