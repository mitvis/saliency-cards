# Attribute Examples

Saliency cards break saliency methods into 10 user-centric attributes that are important when choosing a saliency method. 

In this directory, we provide examples for eight of the attributes (shown in Figure 3 of [Saliency Cards: A Framework to Characterize and Compare Saliency Methods](https://arxiv.org/abs/2206.02958)). We do not include examples of semantic directness (because it is purely descriptive) and input sensitivity (see [Saliency Cards]((https://arxiv.org/abs/2206.02958)) for an example).

## Saliency Card Attributes
### Methodology --- *how the saliency is calculated*
* **Determinism**: measures if a saliency method will always produce the same saliency map given a particular input,label, and model.

* **Hyperparameter Dependence**:
measures a saliency method’s sensitivity to user-specified parameters.

* **Model Agnosticism**:
measures how much access to the model a saliency method requires.

* **Computational Efficiency**:
measures how computationally intensive it is to produce the saliency map.

* **Semantic Directness**:
represents the complexity of this abstraction.


### Sensitivity --- *the relationship between the saliency and the underlying model and data*
* **Input Sensitivity**: measures if a saliency method accurately reflects the model’s sensitivity to transformations in the input space

* **Label Sensitivity**:
measures the saliency method’s response to changes to the target label.

* **Model Sensitivity**:
measures if the output of a saliency method is sensitive to meaningful changes to the model parameters.


### Perceptibility --- *how an end user ultimately interprets the saliency*
* **Minimality**: measures how many unnecessary features are given a significant value in the saliency map.

* **Perceptual Correspondence**:
measures if the perceived signal in the saliency map accurately reflects the feature importance.
