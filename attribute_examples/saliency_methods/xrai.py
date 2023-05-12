"""
XRAI interpretability method class.
Adapted from Google PAIR: https://github.com/PAIR-code/saliency
Original paper: https://arxiv.org/abs/1906.02825
"""

import numpy as np
import saliency.core as saliency
import torch

from .saliency_method import SaliencyMethod
from .vanilla_gradients import VanillaGradients


class XRAI(SaliencyMethod):
    """XRAI saliency method."""

    def __init__(self, model):
        """Extends base method to include XRAI."""
        super().__init__(model)
        self.method = saliency.XRAI()

    def get_saliency(self, input_batch, target_classes=None):
        """Extends base method to compute XRAI attributions."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        def call_model_function(x_value_batch, call_model_args=None,
                                expected_keys=None):
            target_classes = call_model_args['target_classes']
            x_value_batch = x_value_batch.transpose(0, 3, 1, 2) # model: channel first
            input_batch = torch.from_numpy(x_value_batch).to(self.device)

            vanilla_gradients = VanillaGradients(self.model)
            gradients = vanilla_gradients.get_saliency(input_batch,
                                                       target_classes,
                                                       False)
            gradients = gradients.transpose(0, 2, 3, 1) # XRAI: channel last
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

        xrais = []
        for i, single_input in enumerate(input_batch):
            single_input = single_input.detach().cpu().numpy()
            single_input = single_input.transpose(1, 2, 0) # XRAI: channel last
            xrai = self.method.GetMask(single_input, call_model_function,
                                       {'target_classes': target_classes[i]})
            xrai = np.expand_dims(xrai, axis=0)
            xrais.append(xrai)
        return np.array(xrais)
