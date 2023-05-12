"""
LIME interpretability method class.
Original paper: https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf
"""

from lime import lime_image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .saliency_method import SaliencyMethod


class LIME(SaliencyMethod):
    """LIME saliency method."""

    def __init__(self, model):
        """Extends base method to include LIME."""
        super().__init__(model)
        self.method = lime_image.LimeImageExplainer()
        self.num_channels = None # updated when computing saliency

    def get_saliency(self, input_batch, target_classes=None, num_samples=1000,
                     positive_only=True):
        """
        Extends base method to compute LIME attributions.
        
        Additional Args:
        num_samples (int): The number of samples to use to train the surrogate
            model with. Defaults to 1000.
        positive_only (bool): If True, only returns attribution in support of 
            the target class. If False, returns all atrribution. Defaults to True.
        """
        batch_size, num_channels, height, width = input_batch.shape
        self.num_channels = num_channels
        masks = np.empty((batch_size, 1, height, width))
        top_labels = 1
        if target_classes is not None:
            top_labels = None

        input_batch = input_batch.detach().cpu()
        for i, instance in enumerate(input_batch):
            labels = None
            if target_classes is not None:
                labels = [target_classes[i]]
            explanation = self.method.explain_instance(
                instance.numpy().transpose(1, 2, 0),
                self._batch_predict,
                labels=labels,
                top_labels=top_labels,
                hide_color=0,
                num_samples=num_samples,
            )
            if labels is None:
                label = explanation.top_labels[0]
            else:
                label = labels[0]
            masks[i] = self._get_map(explanation, label,
                                     positive_only=positive_only)
        return masks

    def _batch_predict(self, input_batch):
        """Batch predict function required by LIME."""
        self.model = self.model.to(self.device)
        input_batch = torch.from_numpy(input_batch.transpose(0, 3, 1, 2))
        if self.num_channels == 1:
            input_batch = input_batch[:, 0:1, :, :]
        input_batch = input_batch.to(self.device)
        output = self.model(input_batch)
        probabilities = F.softmax(output, dim=1)
        return probabilities.detach().cpu().numpy()

    def _get_map(self, explanation, label, positive_only):
        """Use LIME explanation internals to get feature-level attribution."""
        segments = explanation.segments
        mask = np.zeros(segments.shape)
        feature_explanations = explanation.local_exp[label]
        for feature, saliency in feature_explanations:
            if positive_only:
                saliency = max(saliency, 0)
            mask[segments == feature] = saliency
        return np.expand_dims(mask, axis=0)