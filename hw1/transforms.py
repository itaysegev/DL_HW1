import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        # TODO: Use Tensor.view() to implement the transform.
        # ====== YOUR CODE: ======
        return tensor.view(*self.view_dims)
        # ========================


class InvertColors(object):
    """
    Inverts colors in an image given as a tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) for values in the range [0, 1],
            representing an image.
        :return: The image with inverted colors.
        """
        # TODO: Invert the colors of the input image.
        # ====== YOUR CODE: ======
        max_value = torch.max(x)
        inverted_image = max_value - x
        return inverted_image
        # ========================


class FlipUpDown(object):
    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) representing an image.
        :return: The image, flipped around the horizontal axis.
        """
        # TODO: Flip the input image so that up is down.
        # ====== YOUR CODE: ======
        flipped_image = torch.flip(x, dims=[1])
        return flipped_image
        # ========================


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Prepends an element equal to
    1 to each sample in a given tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A pytorch tensor of shape (D,) or (N1,...Nk, D).
        We assume D is the number of features and the N's are extra
        dimensions. E.g. shape (N,D) for N samples of D features;
        shape (D,) or (1, D) for one sample of D features.
        :return: A tensor with D+1 features, where a '1' was prepended to
        each sample's feature dimension.
        """
        assert x.dim() > 0, "Scalars not supported"

        # TODO:
        #  Add a 1 at the beginning of the given tensor's feature dimension.
        #  Hint: See torch.cat().
        # ====== YOUR CODE: ======
        # Take the shapes of the original tensor
        ones_tensor_shape = list(x.shape)

        # Set the first dim as 1, as we want to concat one element in the first dimension
        ones_tensor_shape[-1] = 1

        # Create a tensor in the shape of the original one, where the first dim is of size one containing only ones
        ones_tensor = torch.ones(tuple(ones_tensor_shape), dtype=x.dtype)

        # Concatenate the tensors
        biased_tensor = torch.cat((ones_tensor, x), dim=-1)

        return biased_tensor
        # ========================
