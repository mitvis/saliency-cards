# **Grad-CAM** Saliency Card
Grad-CAM is a model-dependent, gradient-based method to explain convolutional neural networks (CNNs).

## Methodology
Grad-CAM identifies continuous input regions that are important to the model's output towards the target class. It computes feature importance by extracting the feature maps from an intermediate convolutional layer (typically the last convolutional layer) and weighting them by the gradient of the target output with respect to that layer. The weighted feature maps are summed to obtain a single map, passed through a ReLU function to remove negatively contributing values, and upsampled to the original input dimensions.

**Developed by:** Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra at Georgia Institute of Technology.

**References:** 
- *Original Paper*: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)

**Implementations and Tutorials:**
- *Original GitHub Repository*: [ramprs/grad-cam](https://github.com/ramprs/grad-cam/)
- *PyTorch Integration via Captum*: [Captum Grad-CAM](https://captum.ai/api/layer.html#gradcam)
- *Keras Integration*: [Keras Grad-CAM Tutorial](https://keras.io/examples/vision/grad_cam/)

**Example:** The Grad-CAM saliency map (right) on an [ImageNet](https://www.image-net.org/) image for the class `boxer` (left) using a [VGG-16](https://arxiv.org/pdf/1409.1556.pdf). This example is from [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)

<img src="gradcam_example.png" alt="Example of Grad-CAM on an image of a dog. The saliency is brightest in on the dog's face." width="400" />

### Determinism
Grad-CAM is deterministic.

### Hyperparameter Dependence
Grad-CAM relies on two hyperparameters: the `interpolation method` and the `convolutional layer`.
* The `interpolation method` upsamples the feature map into the input feature dimensions.
* The `convolutional layer` determines which feature maps to use. Typically, the last convolutional layer is used, but other layers can be used to understand earlier model behavior.

### Model Agnosticism
Grad-CAM requires a differentiable model with convolutional layers and access to the gradients.

### Computational Efficiency
Computing Grad-CAM takes on the order of 1e-2 seconds using the [Captum implementation](https://captum.ai/api/layer.html#gradcam) on a 224x224x3 dimensional [ImageNet](https://www.image-net.org/) image, [ResNet50](https://arxiv.org/abs/1512.03385) model, and one NVidia G100 GPU.

### Semantic Directness
Grad-CAM outputs the positive attributions of the gradient-weighted feature maps from an internal convolutional layer. Interpreting it correctly likely requires an understanding of convolutional models and model gradients.

## Sensitivity Testing

### Input Sensitivity

&#129000; **[Deletion](https://arxiv.org/pdf/1806.07421.pdf)**: The model's output changes as input features are iteratively removed based on their Grad-CAM saliency rank. When evaluated on ResNet50 and VGG16 ImageNet models, Grad-CAM performs better than sliding window saliency but worse than RISE and LIME. In [subsequent evaluations](https://download.arxiv.org/pdf/2201.13291v3.pdf) on ResNet50 and CUB-200-2011, Grad-CAM performs similarly to RISE and worse than Ablation CAM, Grad-CAM++, and Score-CAM.

&#129000; **[Insertion](https://arxiv.org/pdf/1806.07421.pdf)**: The model's output changes as input features are iteratively added based on their Grad-CAM saliency rank. When evaluated on ResNet50 and VGG16 ImageNet models, Grad-CAM performs better than sliding window saliency but worse than RISE. Grad-CAM performs worse LIME using the ResNet50 model and on-par with LIME using the VGG16. In [subsequent evaluations](https://download.arxiv.org/pdf/2201.13291v3.pdf) on ResNet50 and CUB-200-2011, Grad-CAM performs worse than RISE, Ablation CAM, Grad-CAM++, and Score-CAM.

### Label Sensitivity

&#128994; **[Data Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Grad-CAM saliency values change between the original model and a model trained with random permutations.  Evaluated on MNIST and Fashion MNIST using CNN and MLP models.

&#128994; **[Model Contrast Score](https://arxiv.org/pdf/1907.09701.pdf)**: Grad-CAM acheives the highest model contrast score compared to vanilla gradients, SmoothGrad, integrated gradients, integrated gradients with SmoothGrad, guided backpropagation, and guided Grad-CAM. Evaluated on the BAM image dataset.

### Model Sensitivity

&#128994; **[Cascading Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Grad-CAM saliency changes as model layers are successively randomized. At complete randomization, the saliency looks random. Evaluated on an ImageNet Inception V3.

&#128994; **[Independent Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Grad-CAM saliency changes as model layers are independently randomized. At complete randomization, the saliency looks random. Evaluated on an ImageNet Inception V3.

&#128994; **[Model Weight Randomization](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM saliency is dependent on model parameters and significantly differs between a fully trained and fully randomized model. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#129000; **[Repeatability](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM saliency has repeatability similar/slightly better than to a random baseline. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#129000; **[Reproducibility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM saliency differs slightly between two models with different architectures trained on the same data. The saliency maps are more similar than they are different but perform worse than a segmentation model.

## Perceptibility Testing

### Minimality

&#129000; **[Sparsity](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: The ratio between Grad-CAM's maximum and minimum saliency values was 5.28. It had lower sparsity than  Ablation-CAM, Grad-CAM++, RISE, and Score-CAM. Evaluated on a ResNet50 model and CUB-200-2011 dataset.

### Perceptual Correspondence

&#128997; **[Localization Utility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM fails the localization utility test. Its saliency values have less overlap with the ground truth than a random model. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#128997; **[Luminosity Calibration](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: Grad-CAM salieny values reflect the impact on class score as much as random values. Evaluated on a ResNet50 model and CUB-200-2011 dataset.

&#129000; **[Mean IoU](https://www.nature.com/articles/s42256-022-00536-x)**: Grad-CAM's saliency overlaps the most with the ground truth pathologies but performs worse than a human benchmark.

## Citation

```
@inproceedings{grad-cam,
  author    = {Ramprasaath R. Selvaraju and
               Michael Cogswell and
               Abhishek Das and
               Ramakrishna Vedantam and
               Devi Parikh and
               Dhruv Batra},
  title     = {Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  booktitle = {International Conference on Computer Vision ({ICCV})},
  pages     = {618--626},
  publisher = {{IEEE} Computer Society},
  year      = {2017},
}
```
