"""Utility functions to run and visualize interpretability methods."""

import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_saliency(batch, images=None, percentile=99, scheme='grayscale', 
                       absolute_values=True, image_cmap=None, overlay='overlay'):
    """
    Visualizes saliency as 2D grayscale images, 2D heatmap images, 3D 
    overlaid heapmaps, or 3D masked images.

    Args:
    batch (4D array (batch, channels, height, width)): Saliency to visualize.
    images (None or 4D array): Images to overlay the saliency on. If None,
        then 2D saliency is plotted without the image.
    percentile: integer in range 0 to 100. Values above the percentile are
        clipped to 1. Helps visualize the important features.
    scheme ('greyscale' or 'heatmap'): the colorscheme to plot with.
    absolute_values (bool): Whether to use the magnitude of the saliency or not.
    image_cmap (color map string): The color map to color the images with.
    overlay ('heatmap' or 'mask'): Whether to overlay the saliency as a heatmap 
        or mask the image. Will only be used if images is not None.

    Returns: Matplotlib figure of the saliency.
    """
    if absolute_values is True:
        batch = np.abs(batch)
    saliency = _clip_saliency(_normalize_0to1(batch), percentile)
    if scheme == 'grayscale':
        cmap = plt.cm.gray
    elif scheme == 'heatmap':
        cmap = 'jet'
    else:
        raise ValueError(f"Scheme {scheme} is invalid. Must be 'grayscale' or 'heatmap'")
    fig = _plot(saliency, images=images, saliency_cmap=cmap, image_cmap=image_cmap, overlay=overlay)
    return fig


def _plot(batch, images=None, saliency_cmap=plt.cm.gray, image_cmap=None, overlay=True):
    """
    Plot saliency batch.

    Args:
    batch: 3D numpy array (batch, height, width) to visualize.
    images (None or 4D array): Images to overlay the saliency on. If None,
        then saliency is plotted without the image.
    saliency_cmap (color map string): The color map to color the saliency with.
    image_cmap (color map string): The color map to color the images with.
    overlay ('heatmap' or 'mask'): Whether to overlay the saliency as a heatmap 
        or mask the image. Will only be used if images is not None.

    Returns: A matplotlib figure displaying each saliency in the batch.
    """
    batch_size, height, width = batch.shape
    fig, ax = plt.subplots(ncols=batch_size)
    if batch_size == 1:
        ax = [ax]
    plt.axis('off')
    for i, mask in enumerate(batch):
        ax[i].set_axis_off()
        if images is None: # plot just the saliency
            ax[i].imshow(mask, cmap=saliency_cmap, vmin=0, vmax=1)
        else: # overlay the saliency with the images
            image = images[i].transpose(1, 2, 0)
            if overlay == 'overlay':
                ax[i].imshow(image, cmap=image_cmap)
                ax[i].imshow(mask, cmap=saliency_cmap, vmin=0, vmax=1, alpha=0.5)
            elif overlay == 'mask':
                mask = np.expand_dims(mask, -1)
                mask[np.where(mask > 0)] = 1
                mask = np.repeat(mask, 3, axis=-1).astype(int)
                masked_image = mask * image
                ax[i].imshow(masked_image, cmap=saliency_cmap)
            else:
                raise ValueError(f"Overlay is {overlay}. Expected 'overlay' or 'mask'.")
                
    return fig


def _clip_saliency(batch, percentile):
    """
    Clip saliency at given percentile.

    Args:
    batch: 4D numpy array (batch, channels, height, width).
    percentile: integer in range 0 to 100. Values above the percentile are
        clipped to 1.

    Returns: A 4D numpy array scaled between 0 and 1 with all values above
    percentile set to 1.
    """
    batch_size = batch.shape[0]
    saliency = _flatten_saliency(batch)
    vmax = np.percentile(saliency, percentile, axis=(1,2))
    vmax = vmax.reshape((batch_size, 1, 1))
    vmin = np.min(saliency, axis=(1,2))
    vmin = vmin.reshape((batch_size, 1, 1))
    saliency = np.clip((saliency - vmin) / (vmax - vmin), 0, 1)
    return saliency


def _normalize_0to1(batch):
    """
    Normalize a batch such that every value is in the range 0 to 1.

    Args:
    batch: a batch first numpy array to be normalized.

    Returns: A numpy array of the same size as batch, where each item in the
    batch has 0 <= value <= 1.
    """
    axis = tuple(range(1, len(batch.shape)))
    minimum = np.min(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    maximum = np.max(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    normalized_batch = (batch - minimum) / (maximum - minimum)
    return normalized_batch


def _flatten_saliency(batch):
    """
    Flattens saliency by summing the channel dimension.

    Args:
    batch: 4D numpy array (batch, channels, height, width).

    Returns: 3D numpy array (batch, height, width) with the channel dimension
        summed.
    """
    return np.sum(batch, axis=1)
