"""
GradCAM interpretability method class.
Original paper: https://arxiv.org/pdf/1610.02391.pdf
"""

import captum.attr

from .saliency_method import SaliencyMethod


class GradCAM(SaliencyMethod):
    """GradCAM saliency method."""

    def __init__(self, model, layer):
        """
        Extends base method to include GradCAM.

        Additional Args:
        layer: The layer of the model to compute GradCAM with. 
            Must be a convolutional layer.
        """
        super().__init__(model)
        self.method = captum.attr.LayerGradCam(model, layer)

    def get_saliency(self, input_batch, target_classes=None,
                     interpolation_method='bilinear'):
        """
        Extends base method to compute the GradCAM attributions.

        Additional Args:
        interpolation_method (str): method to upsample the GradCAM attributions.
            Can be 'nearest', 'area', 'bilinear', or 'bicubic'. Defaults to
            'bilinear'.
        """
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        gradcam = self.method.attribute(input_batch, target=target_classes)
        output_size = (input_batch.shape[-2], input_batch.shape[-1])
        upsampled_gradcam = captum.attr.LayerAttribution.interpolate(
            gradcam, output_size, interpolate_mode=interpolation_method
        )
        upsampled_gradcam = upsampled_gradcam.detach().cpu().numpy()
        return upsampled_gradcam
