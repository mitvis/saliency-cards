<!-- Adapted from the Huggingface Model Card template: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md -->
---
For more information on saliency cards, see: [Saliency Cards: A Framework to Characterize and Compare Saliency Methods](https://arxiv.org/abs/2206.02958)

# {Method Name} Saliency Card
Provide a brief summary of the saliency method.

## Methodology
Provide a longer summary of how the saliency is computed, its intended use, and important considerations.
- **Developed by:** {developers}
- **Shared by [optional]:** {who is sharing it}
- **References:** {links to relevant papers, blog posts, and demos}
- **Implementations and Tutorials [optional]:** {links to source code, tutorials, and implementations}
- **Aliases [optional]:** {other names the method is referred by}
- **Example:** {a visual example of the method}

### Determinism
Describe the saliency method's sources of non-determinism.

### Hyperparameter Dependence
Describe the saliency method's hyperparameters and provide suggestions on how to set them.

### Model Agnosticism
Describe the saliency method's dependence on specific models.

### Computational Efficiency
Describe the saliency method's computational efficiency and computing expectations.

### Semantic Directness
Describe the saliency method's semantic directness --- i.e., what the saliency map represents and any prior knowledge a user needs to understand the method.

## Sensitivity Testing
Summarize the saliency method's sensitivity to changes in the input, label, and model. Report results on the relevant tests. Use &#128994; to indicate the saliency method passed the evaluation, &#128997; to indicate it failed the evaluation, and &#129000; to indicate the evaluation was inconclusive.

### Input Sensitivity
Describe the saliency method's sensitivity to input changes. Provide the results of the saliency method on input sensitivity tests. Known input sensitivity tests are described below:

[&#128994; / &#129000; / &#128997;] **[Completeness](https://arxiv.org/pdf/1703.01365.pdf)**: Requires the saliency scores to sum to the difference between the model's output on the original input and the model's output on a meaningless input.

[&#128994; / &#129000; / &#128997;] **[Deletion](https://arxiv.org/pdf/1806.07421.pdf)**: Measures the change in the model's output as input features are iteratively removed based on their saliency rank. Subsequent evaluations in: https://download.arxiv.org/pdf/2201.13291v3.pdf.

[&#128994; / &#129000; / &#128997;] **[Faithfulness](https://arxiv.org/pdf/1806.07538.pdf)**: Measures the change in the model's output as input features are obscured or removed based on their saliency rank.

[&#128994; / &#129000; / &#128997;] **[Infidelity](https://arxiv.org/pdf/1901.09392.pdf)**: Measures the mean squared error between the saliency weighted by the input perturbation and the difference in the model's output on the actual input and the perturbed input.

[&#128994; / &#129000; / &#128997;] **[Input Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: Measures the consistency of the saliency when the input features are swapped with synonymous features.

[&#128994; / &#129000; / &#128997;] **[Input Invariance](https://arxiv.org/pdf/1711.00867.pdf)**: Measures the difference between the saliency from a model trained on the original inputs and a model trained on the original inputs with a constant shift.

[&#128994; / &#129000; / &#128997;] **[Insertion](https://arxiv.org/pdf/1806.07421.pdf)**: Measures the change in the model's output as input features are iteratively added based on their saliency rank. Subsequent evaluations in: https://download.arxiv.org/pdf/2201.13291v3.pdf.

[&#128994; / &#129000; / &#128997;] **[Perturbation Testing (LeRF)](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: Measures the change in the model's output as input features are iteratively set to zero starting with features with the lowest saliency.

[&#128994; / &#129000; / &#128997;] **[Perturbation Testing (MoRF)](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: Measures the change in the model's output as input features are iteratively set to zero starting with the features with the highest saliency.

[&#128994; / &#129000; / &#128997;] **[Region Perturbation](https://arxiv.org/pdf/1509.06321.pdf)**: Measures how the model's output changes as input regions are perturbed based on their saliency rank.

[&#128994; / &#129000; / &#128997;] **[ROAR](https://proceedings.neurips.cc/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf)**: Measures the difference in model behavior between a model trained on the original inputs and a model trained on only the salient features from the original model.

[&#128994; / &#129000; / &#128997;] **[Robustness](https://arxiv.org/pdf/1806.08049.pdf)**: Measures the change in saliency when meaningless perturbations are applied to the input features.

[&#128994; / &#129000; / &#128997;] **[Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)**: Measures the change in saliency when insignificant perturbations are added to the input.

[&#128994; / &#129000; / &#128997;] **[Stability](https://arxiv.org/pdf/1806.07538.pdf)**: Measures the change in saliency when adversarial perturbations are added to the input.

[&#128994; / &#129000; / &#128997;] **[Sufficiency](https://arxiv.org/pdf/1810.03805.pdf)**: Tests if the set of salient features is sufficient for the model to make a confident and correct prediction.

### Label Sensitivity
Describe the saliency method's sensitivity to label changes. Provide the results of the saliency method on label sensitivity tests. Known label sensitivity tests are described below:

[&#128994; / &#129000; / &#128997;] **[Data Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Measures the change in saliency between a model trained on the original labels and a model trained with random label permutations.

[&#128994; / &#129000; / &#128997;] **[Model Contrast Score](https://arxiv.org/pdf/1907.09701.pdf)**: Measures the change in saliency between two models trained on controlled variants of the dataset where feature importances are known.

### Model Sensitivity
Describe the saliency method's sensitivity to model changes. Provide the results of the saliency method on model sensitivity tests. Known model sensitivity tests are described below:

[&#128994; / &#129000; / &#128997;] **[Cascading Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Measures how the saliency changes as model weights are successively randomized.

[&#128994; / &#129000; / &#128997;] **[Implementation Invariance](https://arxiv.org/pdf/1703.01365.pdf)**: Tests if the saliency is identical for two models that are functionally equivalent.

[&#128994; / &#129000; / &#128997;] **[Independent Model Parameter Randomization](https://arxiv.org/pdf/1810.03292.pdf)**: Measures how saliency changes as single layers of the model are randomized one at a time.

[&#128994; / &#129000; / &#128997;] **[Linearity](https://arxiv.org/pdf/1703.01365.pdf)**: Tests that the saliency of two composed models is a weighted sum of the saliency for each model.

[&#128994; / &#129000; / &#128997;] **[Model Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: Measures the change in saliency between the original model and its compressed variant.

[&#128994; / &#129000; / &#128997;] **[Model Weight Randomization](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Measures the change in saliency between a fully trained and fully randomized model.

[&#128994; / &#129000; / &#128997;] **[Repeatability](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Measures the difference in saliency between two independently initialized models trained in the same way on the same data.

[&#128994; / &#129000; / &#128997;] **[Reproducibility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Measures the difference in saliency between two models with different architectures trained on the same data.

## Perceptibility Testing
Summarize perceptibility considerations of the saliency method that may impact how it is perceived by an end user. Use &#128994; to indicate the saliency method passed the evaluation, &#128997; to indicate it failed the evaluation, and &#129000; to indicate the evaluation was inconclusive.

### Minimality
Describe the saliency method's minimality. Provide the results of the saliency method on minimality tests. Known minimality tests are described blow:

[&#128994; / &#129000; / &#128997;] **[Minimality](https://arxiv.org/pdf/1810.03805.pdf)**: Tests if the salient features are the smallest set of features the model can use to make a confident and correct prediction.

[&#128994; / &#129000; / &#128997;] **[Sparsity](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: Measures the ratio between the maximum and minimum saliency values. High sparsity means the saliency's values are narrow and focused.

[&#128994; / &#129000; / &#128997;] **[Visual Sharpening](https://arxiv.org/pdf/1706.03825.pdf)**: 
Human evaluation of the "sharpness" of the saliency.

### Perceptual Correspondence
Describe the saliency method's perceptual correspondence. Provide the results of the saliency method on perceptual correspondence tests. Known perceptual correspondence tests are described blow:

[&#128994; / &#129000; / &#128997;] **[Localization Utility](https://pubs.rsna.org/doi/10.1148/ryai.2021200267)**: Measures the intersection of the saliency and the ground truth features.

[&#128994; / &#129000; / &#128997;] **[Luminosity Calibration](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: Measures if the relative saliency for two features is equivalent to their relative impact on the model's output.

[&#128994; / &#129000; / &#128997;] **[Mean IoU](https://www.nature.com/articles/s42256-022-00536-x)**: Measures the intersection-over-union of the salient features and a set of ground truth features.

[&#128994; / &#129000; / &#128997;] **[Plausibility](https://arxiv.org/pdf/2104.05824.pdf)**: Measures if the saliency highlights features known to be important to humans.

[&#128994; / &#129000; / &#128997;] **[The Pointing Game](https://arxiv.org/pdf/1608.00507.pdf) / [Hit Rate](https://www.nature.com/articles/s42256-022-00536-x)**: Measures if the highest saliency value is in the set of ground truth features.


## Citation [optional]
If there is a paper or blog post introducing the model, provide the APA and Bibtex information here.

**BibTeX:**
```
```

