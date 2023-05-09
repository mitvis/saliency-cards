# **Integrated Gradients** Saliency Card 
Integrated gradients is a model-dependent, path-attribution saliency method.

## Methodology
Integrated gradients computes feature importance by measuring the difference in feature importance for the actual input and a meaningless `baseline` input. It approximates the integral of the gradient of the target output with respect to the input, interpolating from the `baseline` input to the actual input. 

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
Integrated gradients is deterministic unless the users chooses a non-deterministic `baseline` value. 

### Hyperparameter Dependence
Integrated gradients is sensitive to its `baseline` parameter. The integrated gradients algorithm computes feature importance by interpolating between a meaningless `baseline` input and the actual input, accumulating the gradients at each step. As a result, the feature importance is zero for any features where the baseline and input feature values are the same. 

The all-zero `baseline` is common; however, options include random noise, a blurred version of the input, the inverse of the input, the input with added noise, or the average of multiple baselines. For more information on the `baseline` parameter and suggestions for how to set it, see: [Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/)

### Model Agnosticism
Integrated gradients requires a differentiable model with access to the gradients.

### Computational Efficiency
Computing integrated gradients takes on the order of 1e-1 seconds using the [Captum implementation](https://captum.ai/api/integrated_gradients.html) on a 224x224x3 dimensional ImageNet image, ResNet50 model, and one NVidia G100 GPU.

### Semantic Directness
The output of integrated gradients is the accumulated gradient between the `baseline` value and the true input value. Interpreting its output requires understanding model gradients and the `baseline` hyperparameter.

## Sensitivity Testing

### Input Sensitivity

&#128994; **[Completeness](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients guarantees completeness through its algorithmic definition. Summing the integrated gradients from a meaningless input to the actual input will equal the difference in the model's output between the two inputs.

&#129000; **[Infidelity](https://arxiv.org/pdf/1901.09392.pdf)**: Infidelity was evaluated on MNIST, CIFAR-10, and ImageNet. On MNIST, integrated gradients outperforms vanilla gradients, vanilla gradients with SmoothGrad, guided backpropagation, and guided backpropagation with SmoothGrad. It performs worse than integrated gradients with SmoothGrad and SHAP. On CIFAR-100, integrated gradients performs worse than all other methods. On ImageNet, integrated gradients performs better than vanilla gradients, equivalently to vanilla gradients with SmoothGrad, and worse than integrated gradients with SmoothGrad, guided backpropagation, and guided backpropagation with SmoothGrad.

&#129000; **[Input Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: Integrated gradients stayed more consistent in response to synonymous feature swaps than vanilla gradients and SmoothGrad on the LSTM and QRNN models, but it was outperformed by SmoothGrad using a transformer model. Input consistency was evaluated using number agreement and gender agreement feature swaps using the Syneval and Winobias datasets. 

&#129000; **[Input Invariance](https://arxiv.org/pdf/1711.00867.pdf)**: Integrated gradients can fail input invariance due to its reliance on a `baseline` value. For example, when testing input invariance using a mean shift transformation, integrated gradients with a 0-vector baseline is not input invariant. In contrast, integrated gradients with the black baseline is input invariant. It is possible to select an input transformation that causes integrated gradients to fail input invariance regardless of the baseline value. Input invariance was evaluated using a CNN on MNIST.

&#129000; **[Perturbation Testing (LeRF)](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: Integrated gradients passes the LeRF perturbation tests for experiments with MNIST, CIFAR-10, and IMDB datasets with MLP, CNN, and LSTM models. It achieves worse than random LeRF perturbation scores using an Inception V3 model on ImageNet. This failure is due to integrated gradients fails to correctly predict the sign of the feature importance, causing important features (with negative importance) to be removed first.

&#128994; **[Perturbation Testing (MoRF)](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: Integrated gradients passes all MoRF perturbation tests. The model's output is sensitive to iterative feature perturbation based on integrated gradients' feature rank using MNIST, CIFAR-10, ImageNet, and IMDB datasets with MLP, CNN, Inception V3, and LSTM models.

&#128997; **[ROAR](https://proceedings.neurips.cc/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf)**: Integrated gradients fails the ROAR test. It performs worse than a random assignment of feature importance when applied to a ResNet50 trained on ImageNet.

&#129000; **[Robustness](https://arxiv.org/pdf/1806.08049.pdf)**: Integrated gradients is somewhat sensitive to random noise. Adding noise to an input causes slight changes to the integrated gradients' saliency. However, it performs better than vanilla gradients, input x gradient, LRP, Occlusion, and LIME when applied to CNN models and MNIST data.

&#129000; **[Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)**: Integrated gradients is less sensitive to meaningless perturbations than vanilla gradients and guided backprop. It is more sensitive than smoothed saliency methods (vanilla gradients with SmoothGrad, integrated gradients with SmoothGrad, and guided backprop with SmoothGrad) and SHAP. Integrated gradients' sensitivity was evaluated on MNIST, CIFAR-10, and ImageNet.

&#128994; **[Stability](https://arxiv.org/pdf/1806.07538.pdf)**: Integrated gradients is fairly stable. It's ouputs change minially in response to adversarial perturbations. In evaluations on MNIST, it performs better than LIME and on-par with vanilla gradients, input x gradient, LRP, and Occlusion.


### Label Sensitivity

&#129000; **[Data Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: The integrated gradients values change when the model is trained on perturbed data labels, but the visualizations of the saliency can show input structure. Evaluated on MNIST and Fashion MNIST.

&#129000; **[Model Contrast Score](https://arxiv.org/pdf/1907.09701.pdf)**: Integrated gradients achieves a better-than-random model contrast score, but performs worse than Grad-CAM and SmoothGrad. It is evaluated on the BAM image dataset.


### Model Sensitivity
&#129000; **[Cascading Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: The integrated gradients saliency values randomize as the trained model becomes more randomized. However, the saliency maps show visual similarity between a randomized and trained model and could be misinterpreted. Evaluted on an ImageNet Inception V3 model.

&#128994; **[Implementation Invariance](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients guarantees implementation invariance through its algorithmic definition. It will produce equivalent saliency values for functionally equivalent models.

&#129000; **[Independent Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: The integrated gradients saliency values randomize when the trained model's layers are randomized. However, the saliency maps show visual similarity between a randomized and trained model and could be misinterpreted. Evaluted on an ImageNet Inception V3 model.

&#128994; **[Linearity](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients guarantees linearity through its algorithmic definition.

&#129000; **[Model Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: Integrated gradients stayed more consistent in response to model distillation than vanilla gradients and SmoothGrad on the LSTM and QRNN models, but it was outperformed by SmoothGrad on a transformer model. Evaluated using models distilled via DistillBert applied to number and gender agreement tasks on the Syneval and Winobias datasets. 

&#129000; **[Model Weight Randomization](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: At full model randomization, integrated gradients is almost as random as a random baseline. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#128997; **[Repeatability](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients fails the repeatability test. Its saliency values for two models trained in the same way are more dissimilar than they are similar. Evaluated on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.

&#128997; **[Reproducibility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients fails the reproducibility test. Its saliency values for two models with different architectures trained in the same way and on the same data are more dissimilar than they are similar. Evaluated using Inception V3 and DenseNet-121 on SIIM-ACR Pneumothorax and RSNA Pneumonia medical images.


## Perceptibility Testing

### Minimality
&#129000; **[Visual Sharpness](https://arxiv.org/pdf/1706.03825.pdf)**: Integrated gradients are less visually coherent than a smoothed saliency method.

### Perceptual Correspondence
&#128997; **[Localization Utility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients have less overlap with the ground truth than a random model.

&#128997; **[Mean IoU](https://www.nature.com/articles/s42256-022-00536-x)**: Integrated gradients had lower IoU with the ground truth than CAM and occlusion-based methods. It also had lower IoU with the ground truth than human localization in a chest x-ray setting.

&#128994; **[Plausibility](https://arxiv.org/pdf/2104.05824.pdf)**: Integrated gradients highlights human-important features more than vanilla gradients and SmoothGrad across most architectures.

&#129000; **[The Pointing Game](https://arxiv.org/pdf/1608.00507.pdf)**: Integrated gradients' most salient feature was in the ground truth approximately the same as amount of times as other saliency methods. However, its most salient feature was in the ground truth less than human localization in a chest x-ray setting. Tested by: [Benchmarking saliency methods for chest X-ray interpretation by Saporta et al.](https://www.nature.com/articles/s42256-022-00536-x)

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
