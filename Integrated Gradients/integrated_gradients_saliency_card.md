# **Integrated Gradients** Saliency Card 
Integrated gradients is a model-dependent, path-attribution saliency method.

## Methodology
Integrated gradients computes saliency by comparing the saliency of the actual input to the saliency of a meaningless baseline input. It does so by approximating the integral of the gradient of the target output with respect to the input features, linearly interpolating from the baseline to the actual input.

**Developed by:** Mukund Sundararajan, Ankur Taly, and Qiqi Yan at Google.

**References:** 
- *Original Paper*: [Axiomatic Attribution for Deep Networks](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)
- *Paper on Integrated Gradients Hyperparameters*: [Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/)

**Implementations and Tutorials:**
- *Original GitHub Repository*: [ankurtaly/Integrated-Gradients](https://github.com/ankurtaly/Integrated-Gradients)
- *PyTorch Integration via Captum*: [Captum Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients)
- *TensorFlow Integration*: [TensorFlow Integrated Gradients Tutorial](https://captum.ai/docs/extension/integrated_gradients)

**Aliases:** Path-Integrated Gradients

**Example:** The integrated gradients saliency map (right) on an [ImageNet](https://www.image-net.org/) image of a `cab` (left) using a Pytorch pretrained [ResNet50](https://arxiv.org/abs/1512.03385). This example is computed in `integrated_gradients_example.ipynb`.

<img src="integrated_gradients_example.png" alt="Example of integrated gradients on an image of a taxi cab. The saliency is brightest in the cab region." width="400" />

### Determinism
Integrated gradients is deterministic unless the user chooses a non-deterministic `baseline` value.

### Hyperparameter Dependence
Integrated gradients is sensitive to its `baseline` parameter. Since integrated gradients computes feature importance by integrating from a meaningless `baseline` to the actual input, its saliency is zero for any features where the `baseline` and input values are the same.

The all-zero `baseline` is common; however, other options include random noise, a blurred input, the inverse of the input, the input with added noise, or the average of multiple baselines. For more information on the `baseline` parameter and suggestions for how to set it, see: [Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/)

### Model Agnosticism
Integrated gradients requires a differentiable model with access to the gradients.

### Computational Efficiency
Computing integrated gradients takes on the order of 1e-1 seconds using the [Captum implementation](https://captum.ai/api/integrated_gradients.html) on a 224x224x3 dimensional ImageNet image, ResNet50 model, and one NVidia G100 GPU.

### Semantic Directness
The output of integrated gradients is the accumulated gradient between the `baseline` input and the actual input. Interpreting its output requires understanding model gradients and the impact of the `baseline` hyperparameter.

## Sensitivity Testing

### Input Sensitivity

&#128994; **[Completeness](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients algorithmically guarantees completeness. The sum of the integrated gradients will equal the difference in the model's output between the actual and `baseline` inputs.

&#129000; **[Infidelity](https://arxiv.org/pdf/1901.09392.pdf)**: Integrated gradients' infidelity is inconclusive. Integrated gradients outperforms vanilla gradients on MNIST and ImageNet, performs equivalently to vanilla gradients on CIFAR-100, and performs worse than vanilla gradients with SmoothGrad, guided backpropagation, guided backpropagation with SmoothGrad, and integrated gradients with SmoothGrad across all three datasets.

&#129000; **[Input Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: Integrated gradients' input consistency is inconclusive. It was more consistent than vanilla gradients and SmoothGrad using the LSTM and QRNN models but less consistent than SmoothGrad using a transformer model. Evaluated using number and gender agreement feature swaps with the Syneval and Winobias datasets.

&#129000; **[Input Invariance](https://arxiv.org/pdf/1711.00867.pdf)**: Integrated gradients can fail input invariance due to its `baseline`. A 0-vector `baseline` is not input invariant, while a black baseline is input invariant. Regardless of the baseline, it is possible to selecting an input transformation that guarantees integrated gradients fails input invariance. Evaluated using a CNN on MNIST.

&#129000; **[Perturbation Testing (LeRF)](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: Integrated gradients' LeRF perturbation tests were inconclusive. It passed for experiments using MNIST, CIFAR-10, and IMDB datasets with MLP, CNN, and LSTM models. However, it had worse than random performance using ImageNet Inception V3. This failure is due to incorrectly estimating the sign of the saliency, causing important features (with negative saliency) to be removed first.

&#128994; **[Perturbation Testing (MoRF)](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: Integrated gradients passes all MoRF perturbation tests. Evaluated using MNIST, CIFAR-10, ImageNet, and IMDB datasets with MLP, CNN, Inception V3, and LSTM models.

&#128997; **[ROAR](https://proceedings.neurips.cc/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf)**: Integrated gradients fails the ROAR test. It performs worse than random saliency when applied to an ImageNet ResNet50.

&#129000; **[Robustness](https://arxiv.org/pdf/1806.08049.pdf)**: Integrated gradients is somewhat sensitive to random noise, which causes slight changes to the saliency. However, integrated gradients outperforms vanilla gradients, input x gradient, LRP, Occlusion, and LIME. Evaluated on MNIST CNNs. 

&#129000; **[Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)**: Integrated gradients' sensitivity tests are inconclusive. It is less sensitive to meaningless perturbations than vanilla gradients and guided backpropagation but more sensitive than SmoothGrad applied to vanilla gradients, integrated gradients, and guided backpropagation. Evaluated on MNIST, CIFAR-10, and ImageNet. 

&#128994; **[Stability](https://arxiv.org/pdf/1806.07538.pdf)**: Integrated gradients is relatively stable, and its outputs change minimally in response to adversarial perturbations. It performs better than LIME and equivalently to vanilla gradients, input x gradients, LRP, and Occlusion. Evaluated on MNIST.


### Label Sensitivity

&#129000; **[Data Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Integrated gradients changes appropriately when the model is trained on perturbed data labels. However, its visualizations can misleadingly show input structure. Evaluated on MNIST and Fashion MNIST.

&#129000; **[Model Contrast Score](https://arxiv.org/pdf/1907.09701.pdf)**: Integrated gradients achieves a better-than-random model contrast score but performs worse than Grad-CAM and SmoothGrad. Evaluated on the BAM image dataset.


### Model Sensitivity
&#129000; **[Cascading Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Integrated gradients changes appropriately as the model is progressively randomized. However, its visualizations can misleadingly show input structure. Evaluated an ImageNet Inception V3.

&#128994; **[Implementation Invariance](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients algorithmically guarantees implementation invariance. It will produce equivalent saliency for functionally equivalent models.

&#129000; **[Independent Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Integrated gradients changes appropriately as the model layers are independently randomized. However, its visualizations can misleadingly show input structure. Evaluated an ImageNet Inception V3.

&#128994; **[Linearity](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients algorithmically guarantees linearity. Its saliency on a model composed of two sub-models will equal the weighted sum of its saliency on each sub-model.

&#129000; **[Model Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: Integrated gradients' model consistency is inconclusive. In response to model compression, its saliency stayed more consistent than vanilla gradients and SmoothGrad on LSTM and QRNN models but was less consistent than SmoothGrad on a transformer model. Evaluated using DistillBert distillation and number and gender agreement tasks on the Syneval and Winobias datasets.

&#129000; **[Model Weight Randomization](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: On a randomized model, integrated gradients saliency is near-random. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#128997; **[Repeatability](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients fails repeatability. Its saliency values for two models trained in the same way are more dissimilar than similar. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#128997; **[Reproducibility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients fails reproducibility. Its saliency values for two different architectures trained in the same way are more dissimilar than similar. Evaluated using Inception V3 and DenseNet-121 on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.


## Perceptibility Testing

### Minimality
&#129000; **[Visual Sharpness](https://arxiv.org/pdf/1706.03825.pdf)**: Integrated gradients are less visually coherent than SmoothGrad saliency methods. Evaluated on an ImageNet Inception V3 and MNIST CNN.

### Perceptual Correspondence
&#128997; **[Localization Utility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients fails localization utility. Its saliency values overlap less with the ground truth than a random model. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#128997; **[Mean IoU](https://www.nature.com/articles/s42256-022-00536-x)**: Integrated gradients saliency has lower mean IoU with the ground truth features than CAM and occlusion-based saliency method and human localization. Evaluated using CNNs on CheXpert chest x-ray images.

&#128994; **[Plausibility](https://arxiv.org/pdf/2104.05824.pdf)**: Integrated gradients highlights human-important features more often than vanilla gradients and equivalently to SmoothGrad on LSTM, QRNN, and transformer models. Evaluated on number and gender agreement tasks using the Syneval and Winobias datasets.

&#129000; **[The Pointing Game](https://arxiv.org/pdf/1608.00507.pdf)**: Integrated gradient's most salient feature in the ground truth region as many times as other saliency methods but less than human localization. Evaluated using CNNs on CheXpert chest x-ray images by [Benchmarking saliency methods for chest X-ray interpretation](https://www.nature.com/articles/s42256-022-00536-x).

## Citation
```
@inproceedings{integratedgradients
  author    = {Mukund Sundararajan and Ankur Taly and Qiqi Yan},
  title     = {Axiomatic Attribution for Deep Networks},
  booktitle = {Proceedings of the International Conference on Machine Learning ({ICML})},
  volume    = {70},
  pages     = {3319--3328},
  publisher = {{PMLR}},
  year      = {2017},
}
```
