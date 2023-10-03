import torch


class BaseTransform:
    """
    A class for simple data transformation.
    """
    def __call__(self, data):
        """
        Transforms the data to a PyTorch tensor.

        Args:
            data (numpy.ndarray): The data to transform.

        Returns:
            tensor (torch.Tensor): The transformed data as a PyTorch tensor.

        """
        return torch.tensor(data, dtype=torch.float32)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    