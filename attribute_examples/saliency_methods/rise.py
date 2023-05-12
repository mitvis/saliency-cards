"""
RISE interpretability method class. 
Built from: https://github.com/eclique/RISE
Original paper: https://arxiv.org/abs/1806.07421
"""

import torch
import numpy as np
import os

from .saliency_method import SaliencyMethod
from .rise_master import explanations


class RISE(SaliencyMethod):
    """RISE saliency method."""
    
    def __init__(self, model, input_size, num_masks=500):
        """Extends base method to include LIME.
        
        Additional Args:
        input_size (array): The size of the inputs that will be passed to get_saliency.
        num_masks (int): The number of masks to generate. Default 500.
        """
        super().__init__(model)
        self.method = explanations.RISE(model, input_size, 56) # GPU batch is set as in example
        
        maskspath = os.path.expanduser('~/data/masks.npy')
        self.method.generate_masks(N=num_masks, s=8, p1=0.1, savepath=maskspath)

    
    def get_saliency(self, input_batch, target_classes=None):
        """Extends base method to compute RISE attributions."""        
        if target_classes is None:
            target_classes = self.model(input_batch)
            if torch.is_tensor(target_classes):
                target_classes = target_classes.argmax(dim=1)
            else: # numpy array
                target_classes = target_classes.argmax(axis=1)
            target_classes = self._convert_to_numpy(target_classes)
        
        rise_masks = []
        for i, single_input in enumerate(input_batch):
            single_input_batch = single_input.unsqueeze(0)
            rise_mask = self.method(single_input_batch)
            rise_mask = self._convert_to_numpy(rise_mask)
            rise_mask = rise_mask[target_classes[i]]
            if len(target_classes) == 1:
                rise_mask = np.expand_dims(rise_mask, axis=0)            
            rise_masks.append(rise_mask)
            
        rise_masks = np.expand_dims(np.array(rise_masks), axis=1)
        return rise_masks
    
    def _convert_to_numpy(self, vector):
        """Converts vector to numpy arrays. Useful when using SKLearn models."""
        if isinstance(vector, np.ndarray):
            return vector
        elif torch.is_tensor(vector):
            return vector.detach().cpu().numpy()
        else:
            raise ValueError('Unknown vector type. Vector is not ndarray or tensor.')
    
    