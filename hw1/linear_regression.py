import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======

        y_pred = X.dot(self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        w_opt = None
        # ====== YOUR CODE: ======
        n_samples, n_features = X.shape

        # Calculate optimal weights using closed-form solution
        reg_term = self.reg_lambda * np.eye(n_features)
        w_opt = np.linalg.inv(X.T.dot(X) + reg_term).dot(X.T).dot(y)
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======

    # If feature names are not provided, use all columns except target
    if feature_names is None:
        feature_names = [col for col in df.columns if col != target_name]

    # Extract features and target as numpy arrays
    X = df[feature_names].values
    y = df[target_name].values

    # Fit model and make predictions
    y_pred = model.fit(X, y).predict(X)
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        N, D = X.shape
        ones = np.ones((N, 1))
        xb = np.hstack((ones, X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======

        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X_transformed = poly.fit_transform(X)
        # computes the correlation matrix between the new features
        corr = np.abs(np.corrcoef(X_transformed, rowvar=False))
        np.fill_diagonal(corr, 0)
        # discards any features that have no correlation with any other features
        drop_cols = np.where(np.sum(corr, axis=1) == 0)[0]
        X_transformed = np.delete(X_transformed, drop_cols, axis=1)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    # Calculate correlation coefficients with target feature
    corr_with_target = df.corrwith(df[target_feature])

    # Sort correlations in descending order
    sorted_corr = corr_with_target.abs().sort_values(ascending=False)

    # Select top n features
    top_n_features = sorted_corr[1:n + 1].index.tolist()
    top_n_corr = sorted_corr[1:n + 1].tolist()

    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    n = y.shape[0]
    squared_errors = np.square(y - y_pred)
    mse = np.sum(squared_errors) / n
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    mean_y = np.mean(y)
    ss_total = np.sum(np.square(y - mean_y))
    ss_res = np.sum(np.square(y - y_pred))
    r2 = 1 - (ss_res / ss_total)

    # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    param_grid = {
        'bostonfeaturestransformer__degree': degree_range,
        'linearregressor__reg_lambda': lambda_range
    }
    hyperparameters = model.get_params()
    for param, values in hyperparameters.items():
        if param not in param_grid:
            param_grid[param] = values if type(values) is List else [values]
    mse_scorer = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error, greater_is_better=False)
    grid_search = sklearn.model_selection.GridSearchCV(model, param_grid, scoring=mse_scorer, cv=k_folds)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    # ========================

    return best_params
