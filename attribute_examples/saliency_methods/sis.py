"""
Sufficient Input Subset (SIS) interpretability method class.
Built from: https://github.com/gifford-lab/overinterpretation
Original paper: https://arxiv.org/abs/1810.03805
"""

import torch
import numpy as np

from .saliency_method import SaliencyMethod
from .sis_master import imagenet_backselect


class SIS(SaliencyMethod):
    """SIS saliency method."""
    
    def __init__(self, model):
        """Extends base method to include LIME."""
        super().__init__(model)
    
    def get_saliency(self, input_batch, target_classes=None, sis_threshold=0.85, 
                     features_removed_per_iteration=100):
        """
        Extends base method to compute SIS attributions.
        
        Additional Args:
        sis_threshold (float between 0--1): The SIS model confidence threshold.
        features_removed_per_iteration (int): Number of features to mask every iteration.
        positive_only (bool): If True, only returns attribution in support of 
            the target class. If False, returns all atrribution. Defaults to True.
        """
        cuda = True if 'cuda' in str(input_batch.device) else False
        backselect_results = imagenet_backselect.run_gradient_backward_selection(
            input_batch,
            self.model,
            features_removed_per_iteration,
            max_iters=None,
            add_random_noise=False,
            cuda=cuda)
        
        sis_saliency = []
        for i, backselect_result in enumerate(backselect_results):
            mask_after_iter = np.where(backselect_result.confidences_over_backselect >= sis_threshold)[0][-1]
            image_saliency = (backselect_result.mask_order >= mask_after_iter).astype(int)
            image_saliency = np.repeat(image_saliency, 3, 0)
            sis_saliency.append(image_saliency)

        return np.array(sis_saliency)