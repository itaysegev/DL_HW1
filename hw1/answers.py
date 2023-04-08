r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing the value of k in the KNN algorithm can lead to improved generalization for unseen data up to a certain point,
beyond which the performance may start to degrade.
When k is very small, the model may overfit to the training data and fail to capture the underlying patterns in the data,
resulting in poor generalization to new data.
On the other hand, when k is very large, the model may oversimplify the decision boundary and treat distant points as equally important as nearby points,
which can also lead to poor generalization.
In general, the optimal value of k depends on the complexity of the underlying data distribution,
the noise level in the data, and the size of the training set.
In practice, a common approach is to perform cross-validation to tune the hyperparameter k and choose the value that gives the best performance on a validation set.

"""

part2_q2 = r"""
**Your answer:**
The main advantage of k-fold CV is that it provides a more reliable estimate of the generalization performance of
the model by simulating the process of training and testing on multiple independent subsets of the data
By using k-fold CV, we can obtain a more accurate and unbiased estimate of the true performance of the model,
as it accounts for the variability and randomness in the data and reduces the risk of overfitting to the train-set.
In contrast, training on the entire train-set with various models and selecting the best model based on train-set accuracy can lead to overfitting to the train-set,
as the model may learn the idiosyncrasies and noise in the train-set that do not generalize well to new data.
Moreover, k-fold CV allows us to evaluate and compare the performance of different models or hyperparameters in a fair and systematic way, as all models are trained and tested on the same data subsets.
This can help us select the best model or hyperparameters that optimize the performance on the unseen data.
The main disadvantage of selecting the best model based on test-set accuracy is that it introduces bias in the model selection process,
as the test set is no longer independent from the training process. By using the test set to select the best model,
we are indirectly optimizing the model to perform well on the test set, which may not generalize well to new data.
K-fold CV allows us to evaluate the performance of the model on multiple independent subsets of the data and obtain a more reliable estimate of the true performance. By using k-fold CV, we are simulating the process of training and testing on multiple independent train-test splits and averaging the performance across them. This can reduce the variability and randomness in the data and provide a more accurate estimate of the generalization performance of the model.
Furthermore, k-fold CV allows us to use all the available data for training and evaluation, without sacrificing the effective sample size for model selection. This can lead to a more robust and generalizable model, as it is trained on more data and has learned the underlying patterns that are more likely to generalize to new data.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
In an ideal residual plot, the residuals should be randomly scattered around the horizontal line at zero.
This indicates that the model is capturing the underlying patterns in the data and the errors are random and not systematically biased.
Based on the residual plots shown above, we can see that the model is not perfectly fitting the data.
The residual plots show some patterns in the errors, particularly for the higher target values.
This suggests that the model may be overfitting to some extent.
Comparing the residual plot for the top-5 features with the final plot after CV,
we can see that the latter has fewer patterns in the errors and the residuals are more randomly scattered around the zero line.
This suggests that the final model after CV is a better fit to the data than the model based on just the top-5 features.
"""

part4_q2 = r"""
**Your answer:**
Adding non-linear features to our data can capture more complex relationships between the independent and dependent variables.
In linear regression, adding non-linear features can allow us to fit curves to the data, rather than just straight lines.
This can improve the model's ability to capture non-linear trends and patterns in the data, leading to better predictions.
However, adding non-linear features does not change the fact that we are still using a linear regression model.
The model is still linear in the sense that the parameters are still multiplied by the features and summed to produce a prediction.
The non-linear features simply allow the model to capture non-linear relationships between the features and the target variable.
In theory, we can fit any non-linear function of the original features by adding non-linear features.
However, in practice, we may need to experiment with different types and combinations of non-linear features to find the ones that work best for our specific problem.
In a linear classification model,adding non-linear features would affect the decision boundary by allowing it to become
more complex and potentially non-linear. The decision boundary would no longer be a hyperplane,as it would be in a linear model with linear features only.
Instead, it would be a more complex boundary that could potentially curve and bend to better separate the different classes. This can improve the model's ability to classify complex and non-linear patterns in the data. However, as with linear regression,
we may need to experiment with different types and combinations of non-linear features to find the ones that work best for our specific classification problem.

"""

part4_q3 = r"""
**Your answer:**
In the above CV code, np.logspace was used instead of np.linspace when defining the range for lambda
because np.logspace creates a range of values that are spaced logarithmically.
This is advantageous for CV because changing the value of lambda by a factor of 10 has a greater impact on the model's performance than changing it by a constant value. By using a logarithmic scale for lambda,
we can explore a wide range of values with relatively few steps and obtain a better sense of the optimal value.
The model was fitted to the data k_folds * len(lambda_range) * len(degree_range) times.
In our case, we have 20 values for lambda and 3 values for degree, giving a total of 60 combinations of hyperparameters.
For each combination, we perform 3-fold cross-validation, so the model is fitted to data 3 times per combination.
Therefore, the model is fitted to data a total of 60 * 3 = 180 times, not including the final fit on the entire training set.

"""

# ==============
