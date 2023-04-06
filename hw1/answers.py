r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

##### Question 1

1.  **False** -  The test set allows us to estimate our out-of-sample error,
    which is a way to estimate our model's performance on an unseen data (new data).
    The in-sample error on the other hand allows us to understand how the model fits the training data,  i.e.
    the "seen" data. It is hence computed on the training dataset.
    
2.  **False** -  Generally speaking, this claim is wrong as we may have a data with an inner-order for example.
    i.e., in the example above, we could have all the images of "horses" placed at the beginning (idx-wise). 
    Assuming, for example, 20% of the dataset are images of "horses", in a split which defines the test-set as the
    first 20% images, we'll get unuseful split in the sense of learning how to label "horse" images (as we didn't 
    see any in our supervised learning).
    So, with that being said, a useful split will be such where both training and test sets are representations 
    of the true distribution of the data (so the model could learn how to generalize unseen data well enough).

3.  **True** -  As was mentioned, the test-set is used to estimate the out-of-sample error. I.e., to estimate
    the model's performance on an unseen data. By using the test-set in the cross-validation process, the test
    set won't be independent of the training data anymore (folds are being used for training and validation), 
    and hence we'll most probably get biased (and optimistic) estimations of our performances. 
    
4.  **False** - As was taught in the 'Intro. to ML' course, we estimate the generalization error by averaging the 
    scores on the validation sets, over all the folds. Hence, the validation-set performance of each fold is not a 
    proxy for the model's generalization error.
    The cross-validation goal is to "enhance" the usage of our data, and hence each validation set in the folds, is 
    used to estimate the generalization error on the training sets of that specific fold.

"""

part1_q2 = r"""
**Your answer:**

##### Question 2
    
Our friend's approach **is not justified**. 

The purpose of the test-set is to estimate the model's performance on an unseen data (as repeatedly explained).
In our friend's approach, he uses the test-set to choose the hyperparameter of the regularization. By doing that, he is 
biasing the estimation. The estimation is expected to be optimistic (overfitting the the test-set) since it is chosen 
based on the scores we get on the test-set. It's, in some manner, contradicts the notion of using the test-set as 
an unseen data.

Exactly for this problem (of tuning hyperparameters), we learned we can use the training data and split it to have
also validation data. This way, we are keeping the test-set data independent as much as possible from the training data,
and by doing that achieving a more reliable estimation of the out-of-sample/generalization error.

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
