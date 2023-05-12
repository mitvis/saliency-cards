"""
Kernel SHAP interpretability method class. 
Original paper: https://arxiv.org/pdf/1705.07874.pdf
"""

import captum
import torch
import numpy as np
import skimage.segmentation

from .saliency_method import SaliencyMethod


class KernelSHAP(SaliencyMethod):
    """Kernel SHAP interpretability method."""
    
    def __init__(self, model):
        """Extends base method to include kernel SHAP."""
        super().__init__(model)
        self.shap_method = captum.attr.KernelShap(model)
        
    
    def get_saliency(self, input_batch, target_classes=None, baseline=None, n_samples=50):
        """
        Extends base method to compute Kernel SHAP attributions.

        Additional Args:
        baseline (None or torch Tensor): A batch of baseline inputs to integrate from.
            The same shape as input_batch. If None, then random noise baseline is used.
        n_samples (int): The number of samples to use.
        """
        if input_batch.shape[0] > 1:
            raise ValueError('Batched computation not possible for Kernel SHAP')
        
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)
            
        if baseline is None:
            baseline = torch.randn(input_batch.shape)
        baseline = baseline.to(self.device)
        
        channel_last_input = input_batch[0, :3, :, :].double().detach().cpu().numpy().transpose(1, 2, 0)
        super_pixels = skimage.segmentation.quickshift(channel_last_input)
        feature_mask = torch.from_numpy(super_pixels).to(self.device)

        shap = self.shap_method.attribute(input_batch,
                                          baselines=baseline,
                                          target=target_classes,
                                          feature_mask=feature_mask,
                                          n_samples=n_samples,
                                          return_input_shape=True,)
        shap = shap.detach().cpu().numpy()
        return shap