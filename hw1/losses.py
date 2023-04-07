import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        true_label_score_vec = x_scores[torch.arange(len(x_scores)), y]

        # Index tensor to select the elements of vec
        idx = torch.arange(len(true_label_score_vec)).reshape(-1, 1)

        # Repeat the true label score for future broadcasting operation
        true_label_score_mat = true_label_score_vec[idx].repeat(1, x_scores.shape[1])

        # Calculate the loss matrix
        M = self.delta + x_scores - true_label_score_mat

        # Zero the loss on the true ground label
        M[torch.arange(len(y)), y] = 0

        # Perform max operation in loss
        M[M < 0] = 0

        loss = M.sum(dim=1).sum() / x.shape[0]
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["M"] = M
        self.grad_ctx["x"] = x
        self.grad_ctx["y"] = y
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # Set vars
        G = self.grad_ctx["M"]
        y = self.grad_ctx["y"]
        x = self.grad_ctx["x"]
        N = self.grad_ctx["x"].shape[0]

        # Check if m_i_j > 0 for getting x_i_j * 1 in the matmul with x.t()
        G[G > 0] = 1

        # For L_i grad w.r.t w_yi (sum as was formulated)
        G[torch.arange(len(y)), y] = -1 * (G.sum(dim=1))

        # Compute final grad
        grad = (1 / N) * torch.matmul(x.t(), G)
        # ========================

        return grad
