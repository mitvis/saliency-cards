# Utility functions to create the Saliency Cards paper results.

import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def load_model_from_pytorch(architecture, pretrained, device, num_classes=1000):
    """Load PyTorch torvhvision model."""
    model = torchvision.models.__dict__[architecture](pretrained=pretrained, num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model

def load_pil_image(image_path):
    """Load PIL image from disk."""
    with PIL.Image.open(image_path) as image:
        image = image.convert('RGB')
        return image
    
def imagenet_transform(normalize=True):
    """Return the ImageNet transform. If normalize, normalize transformation is included."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if normalize is True:
        transform = transforms.Compose([transform, imagenet_normalize()])
    return transform

def imagenet_normalize():
    """Return the ImageNet normalization transform."""
    return transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])