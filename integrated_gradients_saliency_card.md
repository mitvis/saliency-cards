# Saliency Card for Integrated Gradients
Integrated Gradients 

# Methodology

Provide a longer summary of what this saliency method does, its intended use, and important considerations.

**Example:** Integrated Gradients saliency map (right) on an ImageNet image of a cap (left) using a Pytorch pretrained ResNet50.

<img src="https://github.com/aboggust/saliency-cards/blob/integrated-gradients/integrated_gradients_example.png" alt="Example of Integrated Gradients on an image of a taxi cab. The saliency is brightest in the cab region." width="400" />

**Developed by:** Mukund Sundararajan, Ankur Taly, and Qiqi Yan at Google.

**References:** 
- *Original Paper*: [Axiomatic Attribution for Deep Networks by Sundararajan et. al.](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)
- *Paper on setting Integrated Gradients' baseline values*: [Visualizing the Impact of Feature Attribution Baselines by Sturmfels et. al.](https://distill.pub/2020/attribution-baselines/)

## Determinism
Integrated Gradients is fully deterministic, unless the users chooses a non-deterministic baseline value. 

## Hyperparameter Dependence
Integrated Gradients is sensitive to its baseline parameter.
The Intergrated Gradients algorithm computes feature importance by interpolating between a meaningless baseline input and the true input, accumulating the gradients at each step.
As a result, the saliency will be zero for any features where the baseline feature value and input feature value are the same.
So, it is critical to choose an appropriate baseline input that is meaningless in your task. 

For example, a commom practice in image classification tasks, is to use a black baseline (the all zero image) because it should be uninformative to a model trained to classify natural objects.
However, a black baseline can be misleading when working with datasets that where black pixels convey meaning, such as x-ray images where fractures appear as a black line through the bone.
In this setting, integrated gradients with a black baseline will indicate the fracture pixels are not important because the black fracture pixels have the same value as the "meaningless" baseline.

Other baseline options include: random noise, a blurred version of the input, the inverse of the input, the input with added noise, or the average of multiple baselines.
For more information on Integrated Gradients' sensitivity to the baseline parameter and suggestions on how to set it, see: [Visualizing the Impact of Feature Attribution Baselines by Sturmfels et. al.](https://distill.pub/2020/attribution-baselines/).

## Model Agnosticism
Integrated Gradients requires a differentiable model with access to the gradients.

## Computational Efficiency
Running Integrated Gradients take ~1 second for a 224x224x3 dimensional ImageNet image using a ResNet50 model and one NVidia G100 GPU.

# Sensitivity Testing
Summarize the saliency method's sensitivity to changes in the input, label, and model.

## Input Sensitivity
Describe the saliency method's sensitivity to input changes. Provide the results of the saliency method on input sensitivity tests.

## Label Sensitivity
Describe the saliency method's sensitivity to label changes. Provide the results of the saliency method on label sensitivity tests.

## Model Sensitivity
Describe the saliency method's sensitivity to model changes. Provide the results of the saliency method on model sensitivity tests.

# Perceptibility Testing
Summarize perceptibility considerations of tthe saliency method that may impact how it is percieved by an end user.

## Minimality
Describe the saliency method's minimality. Provide the results of the saliency method on minimality tests.

## Perceptual Correspondence
Describe the saliency method's perceptual correspondence. Provide the results of the saliency method on perceptual correspondence tests.

## Semantic Directness
Describe the saliency method's semantic directness. Provide the results of the saliency method on semantic directness tests.

# Citation
If there is a paper or blog post introducing the model, provide the APA and Bibtex information here.

**BibTeX:**
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
