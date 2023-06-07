# LIME Saliency Card
LIME is a model-agnostic saliency method.

## Methodology
LIME trains an interpretable surrogate model to mimic the original model's decision on the input. It perturbs the original input and uses the original model's output on the perturbs inputs to train a simpler interpretable surrogate model. It extracts feature importance directly from the surrogate model. 

**Developed by:** Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin at the University of Washington

**References:**
- *Original Paper*: [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938v1.pdf)
- *Blog Post*: [O'Reilly Blog](https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/)
- *Blog Post*: [Marco Ribeiro blog post on LIME](https://homes.cs.washington.edu/~marcotcr/blog/lime/)

**Implementations and Tutorials:** 
- [Captum implementation](https://captum.ai/api/lime.html)

**Example:** The LIME saliency map (right) on an image for the class `electric guitar` (left) using an [Inception v3](https://arxiv.org/pdf/1512.00567.pdf). This example is from [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938v1.pdf).

<img src="lime_example.png" alt="Example of LIME on an image of a dog playing guitar. The saliency selects the neck of the guitar." width="400" />

### Determinism
LIME perturbs the inputs non-deterministically. 

### Hyperparameter Dependence
LIME is dependent on its `surrogate model` and its parameters, as well as the parameters of its input perturbation procedure.

### Model Agnosticism
LIME is model agnostic.

### Computational Efficiency
Computing LIME takes on the order of $1\mathrm{e}{1}$ seconds using the [Captum implementation](https://captum.ai/api/saliency.html) on a 224x224x3 dimensional [ImageNet](https://www.image-net.org/) image, [ResNet50](https://arxiv.org/abs/1512.03385) model, and one NVidia G100 GPU.

### Semantic Directness
The output of LIME is the positively contributing features learned by a surrogate model trained to mimic the original model’s local decision boundary for the input. It likely requires understanding of model decision boundaries and interpretable models.

## Sensitivity Testing

### Input Sensitivity

&#129000; **[Deletion](https://arxiv.org/pdf/1806.07421.pdf)**: LIME's deletion performance is inconclusive. LIME performs better than sliding window saliency and Grad-CAM but worse than RISE. Evaluated using ResNet50 and VGG16 on ImageNet.

&#128994; **[Faithfulness](https://arxiv.org/pdf/1806.07538.pdf)**: LIME has faithfulness similar to SHAP. The correlation between its saliency and known importance is non-zero. Evaluated on UCI datasets and a CNN model.

&#129000; **[Insertion](https://arxiv.org/pdf/1806.07421.pdf)**: LIME's insertion performance is inconclusive. LIME performs better than sliding window saliency and Grad-CAM but worse than RISE. Evaluated using ResNet50 and VGG16 on ImageNet.

&#128997; **[Robustness](https://arxiv.org/pdf/1806.08049.pdf)**: LIME is highly sensitive to meaningless perturbations. Its superpixel-based saliency take up large feature regions and change drastically as the input is perturbed. Evaluated using an MNIST CNN.

&#128997; **[Stability](https://arxiv.org/pdf/1806.07538.pdf)**: LIME is highly sensitive to adversarial perturbations and is more sensitive than vanilla gradients, input x gradient, integrated gradients, and occlusion. Evaluated using an MNIST CNN.

### Label Sensitivity
Not tested for label sensitvity.

### Model Sensitivity
Not tested for model sensitvity.

## Perceptibility Testing

### Minimality
Not tested for minimality.

### Perceptual Correspondence
Not tested for perceptual correspondence.

## Citation

**BibTeX:**
```
@inproceedings{lime,
    title = {"{W}hy Should I Trust You?": {E}xplaining the Predictions of Any Classifier},
    author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
    pages={1135--1144},
    year={2016},
    booktitle = {Proceedings of the International Conference on Knowledge Discovery and Data Mining ({KDD})},
    publisher = {{ACM}},
}
```

