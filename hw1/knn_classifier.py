import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        x_train = []
        y_train = []
        for data, labels in dl_train:
            x_train.append(data)
            y_train.append(labels)
        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0)
        n_classes = len(torch.unique(y_train))
        # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            # Find indices of k-nearest neighbors for each test sample
            _, indices = torch.topk(dist_matrix[:, i], k=self.k, largest=False)

            # Get labels of k-nearest neighbors for each test sample
            knn_labels = self.y_train[indices]

            # Predict most common class among k-nearest neighbors
            y_pred[i] = torch.mode(knn_labels).values.item()
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist

    dists = None
    # ====== YOUR CODE: ======
    x1_squared = torch.sum(x1 ** 2, dim=1, keepdim=True)
    x2_squared = torch.sum(x2 ** 2, dim=1, keepdim=True)
    x1x2 = torch.matmul(x1, x2.transpose(0, 1))

    dists = torch.sqrt(x1_squared - 2 * x1x2 + x2_squared.transpose(0, 1))
    # ========================

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    accuracy = torch.sum(y_pred == y).item() / y.shape[0]
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        n_samples = len(ds_train)
        fold_size = n_samples // num_folds
        total_acc = 0.0
        acc_list = []
        for j in range(num_folds):
            # Define indices for the current fold
            val_start = j * fold_size
            val_end = (j + 1) * fold_size
            if j == num_folds - 1:
                val_end = n_samples
            val_indices = list(range(val_start, val_end))
            train_indices = list(set(range(n_samples)) - set(val_indices))

            # Split dataset into train and validation sets
            train_set = torch.utils.data.Subset(ds_train, train_indices)
            train_dl = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
            val_set = torch.utils.data.Subset(ds_train, val_indices)
            val_dl = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=2)
            # Create kNN model and train it on the current fold's train set
            model.train(train_dl)

            x_val, y_val = dataloader_utils.flatten(val_dl)
            # Evaluate the model on the current fold's validation set
            y_val_pred = model.predict(x_val)
            acc = accuracy(y_val, y_val_pred)

            acc_list.append(acc_list)

            # Compute average accuracy across all folds for the current k
        accuracies.append(acc_list)

        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
