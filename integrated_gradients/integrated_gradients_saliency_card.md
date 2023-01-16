# Saliency Card for **Integrated Gradients**
Integrated gradients is a model-dependent, path-attribution saliency method that can be used for augmenting accuracy metrics, model debugging, and feature extraction.

# Methodology
Integrated gradients computes feature importance by measuring global changes in the model’s output with respect to the input values. Unlike vanilla gradients that expose local variability in the model’s output in response to input feature changes, integrated gradients consider global changes by accumulating local gradients weighted by their impact on the model’s output.

Integrated gradients is defined as:
$$E_{IG}(I,c) = (I - \bar{I}) x \int_{\alpha=0}^{1}{\frac{\partial S_c(\bar{I} + \alpha (I - \bar{I}))}{\partial I}\partial \alpha}$$
where $\bar{I}$ is the `baseline` hyperparmeter. $\alpha$ integrates from 0 to 1 which scales the input to the model from the baseline to the original input.

Integrated gradients prevents the gradient saturation problem by capturing a larger range of the model's function. However, the choice of baseline can significantly impact the result. Integrating from the baseline value to a similar feature value will be unimportant even if the feature is locally important.

**Developed by:** Mukund Sundararajan, Ankur Taly, and Qiqi Yan at Google.

**References:** 
- *Original Paper*: [Axiomatic Attribution for Deep Networks by Sundararajan et. al.](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)
- *Paper on setting Integrated Gradients' hyperparameters*: [Visualizing the Impact of Feature Attribution Baselines by Sturmfels et. al.](https://distill.pub/2020/attribution-baselines/)

**Implementations and Tutorials:**
- *Original GitHub Repository*: [ankurtaly/Integrated-Gradients](https://github.com/ankurtaly/Integrated-Gradients)
- *PyTorch Integration via Captum*: [Captum Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients)
- *TensorFlow Integration*: [TensorFlow Integrated Gradients Tutorial](https://captum.ai/docs/extension/integrated_gradients)

**Aliases:** Path-Integrated Gradients

**Example:** The integrated gradients saliency map (right) on an [ImageNet](https://www.image-net.org/) image of a `cab` (left) using a Pytorch pretrained [ResNet50](https://arxiv.org/abs/1512.03385).

<img src="integrated_gradients_example.png" alt="Example of integrated gradients on an image of a taxi cab. The saliency is brightest in the cab region." width="400" />


## Determinism
Integrated gradients is fully deterministic, unless the users chooses a non-deterministic baseline value. 

## Hyperparameter Dependence
Integrated gradients is sensitive to its `baseline` parameter. The integrated gradients algorithm computes feature importance by interpolating between a meaningless baseline input and the true input, accumulating the gradients at each step. As a result, the saliency will be zero for any features where the baseline and input feature values are the same. So, choosing an appropriate baseline input that is meaningless in the task is critical. 

For example, a common practice in image classification tasks is to use a black baseline (the all-zero image) because it should be uninformative to a model trained to classify natural objects. However, a black baseline can be misleading when working with datasets where black pixels convey meaning, such as x-ray images where fractures appear as a black line through the bone. In this setting, integrated gradients with a black baseline will indicate that the fracture pixels are unimportant because the black fracture pixels have the same value as the "meaningless" baseline.

Other baseline options include random noise, a blurred version of the input, the inverse of the input, the input with added noise, or the average of multiple baselines. For more information on integrated gradients' dependence on the `baseline` parameter and suggestions on how to set it, see: [Visualizing the Impact of Feature Attribution Baselines by Sturmfels et. al.](https://distill.pub/2020/attribution-baselines/).

## Model Agnosticism
Integrated gradients requires a differentiable model with access to the gradients.

## Computational Efficiency
Running integrated gradients take ~1 second for a 224x224x3 dimensional [ImageNet](https://www.image-net.org/) image using a [ResNet50](https://arxiv.org/abs/1512.03385) model and one NVidia G100 GPU. It would take approximately 16.6 days to extract saliency maps across the entire ImageNet dataset using these settings.

## Semantic Directness
The output of integrated gradients is the accumulated gradient between the baseline value and the true input value. Interpreting its output requires understanding model gradients and the `baseline` hyperparameter.


# Sensitivity Testing

## Input Sensitivity

&#129001; **[Completeness](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients was designed to satisfy completeness.

&#129000; **[Infidelity](https://arxiv.org/pdf/1901.09392.pdf)**: Integrated gradients has less infidelity than vanilla gradients and guided backprop. It is outperformed by smoothed saliency methods (vanilla gradients with SmoothGrad, integrated gradients with SmoothGrad, and guided backprop with SmoothGrad) and SHAP.

&#129001; **[Input Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: More input consistent than vanilla gradients and SmoothGrad across most architectures.

&#128997; **[Input Invariance](https://arxiv.org/pdf/1711.00867.pdf)**: Integrated gradients can fail input invariance due to its reliance on a `baseline` value. For example, when testing input invariance using a mean shift transformation, integrated gradients with a 0-vector baseline is not input invariant. In contrast, integrated gradients with the black baseline is input invariant. It is possible to select an input transformation that causes integrated gradients to fail input invariance regardless of the baseline value.

&#129001; **[Perturbation Testing](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: Integrated gradients passes perturbation testing. The model's output is sensitive to iterative feature perturbation based on integrated gradients' feature rank. Integrated gradients can capture the interactions between multiple features.

&#128997; **[ROAR](https://proceedings.neurips.cc/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf)**: Integrated gradients fails the ROAR test. It performs worse than a random assignment of feature importance.

&#129001; **[Robustness](https://arxiv.org/pdf/1806.08049.pdf)**: Integrated gradients passes robustness tests. Adding noise to an input only causes slight changes to the integrated gradient's saliency.

&#129000; **[Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)**: Integrated gradients is less sensitive to meaningless perturbations than vanilla gradients and guided backprop. It is more sensitive than smoothed saliency methods (vanilla gradients with SmoothGrad, integrated gradients with SmoothGrad, and guided backprop with SmoothGrad) and SHAP.

&#129001; **[Stability](https://arxiv.org/pdf/1806.07538.pdf)**: Integrated gradients' is relatively stable. Its saliency maps change slightly when Gaussian noise is added to the input.


## Label Sensitivity

&#129000; **[Data Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Integrated gradients' values are sensitive to data randomization, but their visual feature maps still show input structure.

&#129001; **[Model Contrast Score](https://arxiv.org/pdf/1907.09701.pdf)**: Integrated gradients saliency maps differ for background and foreground objects. They outperform the model contrast score of a random saliency map.


## Model Sensitivity
&#129000; **[Cascading Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: The integrated gradients' saliency maps randomize as the trained model becomes more randomized. However, the visual saliency maps show visual similarity between a randomized and trained model and could be misinterpreted.

&#129001; **[Implementation Invariance](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients was designed to satisfy implementation invariance.

&#129000; **[Independent Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: The integrated gradients' saliency maps randomize when the trained model's layers are randomized. However, the visual saliency maps show visual similarity between a randomized and trained model and could be misinterpreted.

&#129001; **[Linearity](https://arxiv.org/pdf/1703.01365.pdf)**: Integrated gradients was designed to satisfy implementation invariance.

&#129001; **[Model Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: More model consistent than vanilla gradients and SmoothGrad across most architectures.

&#129000; **[Model Weight Randomization](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: At full model randomization, integrated gradients is almost as random as a random baseline.

&#128997; **[Repeatability](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients for two random initializations of the same model are more dissimilar than they are similar.

&#128997; **[Reproducibility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients for an InceptionV3 and DenseNet-121 trained in the same way are more dissimilar than they are similar.


# Perceptibility Testing

## Minimality
&#129000; **[Visual Sharpness](https://arxiv.org/pdf/1706.03825.pdf)**: Integrated gradients are less visually coherent than a smoothed saliency method.

## Perceptual Correspondence
&#128997; **[Localization Utility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Integrated gradients performed worse than a random model.

&#128997; **[Mean IoU](https://www.nature.com/articles/s42256-022-00536-x)**: Integrated gradients performed worse than CAM and occlusion-based methods. It also performed worse than human localization in a chest x-ray setting.

&#129001; **[Plausibility](https://arxiv.org/pdf/2104.05824.pdf)**: Integrated gradients is more plausible than vanilla gradients and SmoothGrad across most architectures. It passes the benchmark 77.6% of the time.

&#129000; **The Pointing Game**: Integrated gradients performed better or on par with other saliency methods. However, it performed worse than human localization in a chest x-ray setting. Tested by: [Benchmarking saliency methods for chest X-ray interpretation by Saporta et al.](https://www.nature.com/articles/s42256-022-00536-x)

# Citation
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
