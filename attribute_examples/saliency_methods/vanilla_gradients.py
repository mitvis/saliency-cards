"""
Vanilla gradients interpretability method class.
Original papers: https://www.researchgate.net/profile/Aaron_Courville/
publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_
Network/links/53ff82b00cf24c81027da530.pdf and
https://arxiv.org/pdf/1312.6034.pdf
"""

import captum.attr

from .saliency_method import SaliencyMethod


class VanillaGradients(SaliencyMethod):
    """Vanilla Gradients saliency method."""

    def __init__(self, model):
        """Extends base method to include vanilla gradients."""
        super().__init__(model)
        self.method = captum.attr.Saliency(model)

    def get_saliency(self, input_batch, target_classes=None, absolute=True):
        """
        Extends base method to compute vanilla gradients.

        Additional Args:
        absolute: a boolean that if True, returns the magnitude of the
            gradients. If False, returns the gradients and their signs. Defaults
            to True.
        """
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        vanilla_gradients = self.method.attribute(input_batch,
                                                  target=target_classes,
                                                  abs=absolute)
        vanilla_gradients = vanilla_gradients.detach().cpu().numpy()
        return vanilla_gradients
    