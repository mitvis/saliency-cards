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

&#129000; **[Deletion and Insertion](https://arxiv.org/pdf/1806.07421.pdf)**: The model's output changes as input features are iteratively removed based on their Grad-CAM saliency rank. Grad-CAM performs better than sliding window saliency but worse than RISE and LIME.

&#129000; **[Reliability](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: The model's output changes as input features are progressively masked or revealed based on their Grad-CAM rank. Grad-CAM performs similarly to other methods like RISE and Grad-CAM++.

### Label Sensitivity

&#128994; **[Data Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Grad-CAM's saliency map changes between the original model and a model trained with random permutations. Its original saliency map focuses on the main object, and its saliency map after label permutation highlights random, disconnected patches.

&#128994; **[Model Contrast Score](https://arxiv.org/pdf/1907.09701.pdf)**: Grad-CAM's saliency map changes significantly between a model trained with object labels and a model trained with background labels. Of the saliency methods tested, it has the highest model contrast score.

### Model Sensitivity

&#128994; **[Cascading Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Grad-CAM's saliency map changes as model weights are successively randomized. At complete randomization, the saliency looks random.

&#128994; **[Independent Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Grad-CAM's saliency maps changes as model layers are randomized.

&#128994; **[Model Weight Randomization](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM's saliency changes between a fully trained and fully randomized model.

&#129000; **[Repeatability](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM's saliency differs slightly between two independently initialized models trained in the same way on the same data. The saliency maps are more similar than they are different but perform worse than a segmentation model.

&#129000; **[Reproducibility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM's saliency differs slightly between two models with different architectures trained on the same data. The saliency maps are more similar than they are different but perform worse than a segmentation model.

## Perceptibility Testing

### Minimality

&#128997; **[Sparsity](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: Grad-CAM's ratio between the maximum and minimum saliency values was 5.28. It performed worst out of Ablation-CAM, Grad-CAM++, RISE, and Score-CAM.

### Perceptual Correspondence

&#129000; **[Localization Utility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Grad-CAM's saliency maps overlapped with the ground truth more than a random model. However, Grad-CAM's saliency overlapped with the ground truth less than the average ground truth region.

&#129000; **[Luminosity Calibration](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: Grad-CAM's luminosity calibration is similar to other methods tested: Ablation CAM, Grad-CAM++, and RISE. There is no clear correlation between the features' impact and their relative saliency values.

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
