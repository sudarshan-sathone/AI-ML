Day 8 Report — End-to-End Pipeline
🔧 Technical Summary

Today felt like putting everything together into one complete flow instead of working on isolated steps.

I worked on building an end-to-end machine learning pipeline using the California Housing dataset. Unlike earlier days where the focus was on small parts like preprocessing or training, this time the goal was to connect everything — from loading real-world data to evaluating the model.

I started by fetching the dataset directly using fetch_california_housing and converting it into a DataFrame to make it easier to work with. The dataset had multiple features like median income, house age, population, etc., along with the target variable which represents house prices.

After that, I split the data into training and testing sets. This step felt important because it ensures that the model is evaluated on unseen data rather than what it has already learned.

Next, I applied StandardScaler to normalize the feature values. This was necessary because the dataset contains features with very different ranges, and scaling helps the model treat them equally.

Once the data was prepared, I trained a Linear Regression model. The model essentially tries to find a relationship between all input features and the house prices.

After training, I made predictions on the test data and evaluated the model using Mean Absolute Error (MAE) and R² score. I got an MAE of around $53,000 and an R² score of 0.58, which indicates moderate performance.

Finally, I visualized the model’s performance using a residual plot. This helped in understanding how the prediction errors are distributed rather than just relying on numerical metrics.

Overall, this felt like the first time I built something that resembles a real-world ML workflow instead of just experimenting with individual steps.

🐞 Bug Log

There were no major errors, but a few small things needed attention.

One issue was making sure that scaling was applied correctly. Initially, it’s easy to accidentally use fit_transform on both training and testing data, which can lead to data leakage. I ensured that the scaler was fitted only on the training data and then applied to the test data.

Another minor thing was related to understanding the output of the model. Since the target variable is in units of 100,000 dollars, I had to multiply the MAE value to convert it into actual dollar amounts. Without that, the error value would be misleading.

Apart from that, the pipeline ran smoothly.

💭 Conceptual Reflection

If the residual plot shows a clear U-shaped pattern instead of random dots, what does that tell us about the Linear model?

If the residual plot shows a U-shaped pattern, it indicates that the relationship between the features and the target variable is non-linear.

This means that the Linear Regression model is not able to capture the underlying pattern properly, leading to systematic errors. In such cases, the model is underfitting, and a more complex model such as polynomial regression or tree-based methods would be more appropriate.

🧠 Final Thought

Today made it clear that building a model is not just about training it and checking accuracy.

The residual plot part especially stood out, because it showed that even if the model gives decent numerical results, there can still be patterns in the errors that indicate deeper issues.

It also made me realize that real-world data is rarely perfectly linear, and simple models have their limitations.

This felt less like running code and more like actually analyzing how a model behaves.