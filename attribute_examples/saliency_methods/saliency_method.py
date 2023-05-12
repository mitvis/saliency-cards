"""
Parent class for all saliency methods.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch


class SaliencyMethod(ABC):
    """Base class for all saliency methods."""

    def __init__(self, model):
        """Initializes the method with a model and device.

        Args:
        model (PyTorch or SkLearnmodel): The model to compute saliency with.
        """
        self.model = model
        try: # put pytorch models in eval mode
            self.model.eval()
        except: pass # ignore for sklearn-based models
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

    @abstractmethod
    def get_saliency(self, input_batch, target_classes=None):
        """
        Returns activations for the given input. Implemented in subclasses.

        Args:
        input_batch (torch Tensor): The inputs batch with dimensions
            (batch, channel, height, width) to run the saliency method on.
        target_classes (None or int array): None or list of integers 
            indicating the target class for each input. If None, then the 
            predicted classes are used as the target classes.

        Returns:
        A numpy array the same shape as the input_batch of interpretability
        activations with respect to the target classes. Output is not normalized
        or in absolute values.
        """
        pass

    def get_saliency_smoothed(self, input_batch, target_classes=None,
                              std_spread=.15, num_samples=25, magnitude=True):
        """
        Returns SmoothGrad activations for the given input.

        Args:
        input_batch (torch Tensor): The inputs batch with dimensions
            (batch, channel, height, width) to run the saliency method on.
        target_classes (None or int array): None or list of integers 
            indicating the target class for each input. If None, then the 
            predicted classes are used as the target classes.
        std_spread (float): The amount of noise to add to the input as a
            fraction of the total spread.
        num_samples (int): The number of samples to average across.
        magnitude (bool): Whether to compute sum of squares.

        Returns:
        A numpy array the same shape as input_batch of SmoothGrad
        interpretability activations with respect to the target classes.
        Output is not normalized or in absolute values.

        """
        std = float(
            std_spread * (torch.max(input_batch) - torch.min(input_batch))
        )
        total_gradients = np.zeros(input_batch.shape)
        for _ in range(num_samples):
            noise = torch.empty(input_batch.shape).normal_(mean=0, std=std)
            noise = noise.to(self.device)
            noisy_input_batch = input_batch + noise
            gradients = self.get_saliency(noisy_input_batch,
                                          target_classes=target_classes)
            if magnitude:
                total_gradients += (gradients * gradients)
            else:
                total_gradients += gradients
        output = total_gradients / num_samples
        return output
