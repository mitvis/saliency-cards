"""
Integrated gradients interpretability method class.
Original paper: https://arxiv.org/pdf/1703.01365.pdf
"""

import captum.attr
import torch

from .saliency_method import SaliencyMethod


class IntegratedGradients(SaliencyMethod):
    """Integrated Gradients interpretability method."""

    def __init__(self, model):
        """Extends base method to include integrated gradients."""
        super().__init__(model)
        self.method = captum.attr.IntegratedGradients(self.model)

    def get_saliency(self, input_batch, target_classes=None, baseline=None,
                     num_points=50):
        """
        Extends base method to compute integrated gradients attributions.

        Additional Args:
        baseline (None or torch Tensor): A batch of baseline inputs to integrate from.
            The same shape as input_batch. If None, then random noise baseline is used.
        num_points (int): The number of points to integrate from the baseline to
            the input.
        """
        if baseline is None:
            baseline = torch.randn(input_batch.shape)
        baseline = baseline.to(self.device)

        if target_classes is None: # compute with respect to the predicted class
            target_classes = self.model(input_batch).argmax(dim=1)

        integrated_gradients = self.method.attribute(input_batch,
                                                     baselines=baseline,
                                                     target=target_classes,
                                                     n_steps=num_points)
        integrated_gradients = integrated_gradients.detach().cpu().numpy()
        return integrated_gradients
