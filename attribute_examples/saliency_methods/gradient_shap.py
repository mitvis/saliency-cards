"""
Gradient SHAP interpretability method class.
Original paper: https://arxiv.org/pdf/1705.07874.pdf
"""

import captum
import torch

from .saliency_method import SaliencyMethod


class GradientSHAP(SaliencyMethod):
    """Gradient SHAP saliency method."""

    def __init__(self, model):
        """Extends base method to include Gradient SHAP."""
        super().__init__(model)
        self.method = captum.attr.GradientShap(model)

    def get_saliency(self, input_batch, target_classes=None, baselines=None,
                     num_samples=5):
        """
        Extends base method to compute Gradient SHAP attributions.

        Additional Args:
        baselines (None or torch Tensor): The baslines used to compute attributions.
            If None, uses a the all zero and all ones baselines.
        num_samples (int): Number of noisy inputs to compute.
        """
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        if baselines is None:
            baselines = torch.cat([input_batch * 0, input_batch * 1])
        baselines = baselines.to(self.device)

        shap = self.method.attribute(input_batch,
                                     baselines=baselines,
                                     target=target_classes,
                                     n_samples=num_samples)
        shap = shap.detach().cpu().numpy()
        return shap
