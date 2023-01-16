<!-- Adapted from the Huggingface Model Card template: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md -->
---
For more information on saliency cards, see: [Saliency Cards: A Framework to Characterize and Compare Saliency Methods](https://arxiv.org/abs/2206.02958)

# Saliency Card for **{method name}**
Provide a brief summary of the saliency method.

# Methodology
Provide a longer summary of what this saliency method does, its intended use, and important considerations.
- **Developed by:** {developers}
- **Shared by [optional]:** {who is sharing it}
- **References:** {links to relevant papers, blog posts, and demos}
- **Implementations and Tutorials [optional]:** {links to source code, tutorials, and implementations}
- **Aliases [optional]:** {other names the method is referred by}
- **Example:** {a visual example of the method}

## Determinism
Describe the saliency method's sources of non-determinism.

## Hyperparameter Dependence
Describe the saliency method's hyperparameters and provide suggestions on how to set them.

## Model Agnosticism
Describe the saliency method's dependence on specific models --- for instance, expectations of architecture types of differentiability.

## Computational Efficiency
Describe the saliency method's computational efficiency and computing expectations.

## Semantic Directness
Describe the saliency method's semantic directness --- i.e., how easy interpretable it is to understand the output and any prior knowledge a user needs to understand the method.

# Sensitivity Testing
Summarize the saliency method's sensitivity to changes in the input, label, and model. Report results on the relevant tests.

## Input Sensitivity
Describe the saliency method's sensitivity to input changes. Provide the results of the saliency method on input sensitivity tests. Known input sensitivity tests are described below:

**[Completeness](https://arxiv.org/pdf/1703.01365.pdf)**: 

**[Deletion and Insertion](https://arxiv.org/pdf/1806.07421.pdf)**: 

**[Faithfulness](https://arxiv.org/pdf/1806.07538.pdf)**: 

**[Infidelity](https://arxiv.org/pdf/1901.09392.pdf)**: 

**[Input Consistency](https://arxiv.org/pdf/2104.05824.pdf)**: 

**[Input Invariance](https://arxiv.org/pdf/1711.00867.pdf)**: 

**[Perturbation Testing](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)**: 

**[Region Perturbation](https://arxiv.org/pdf/1509.06321.pdf)**: 

**[Reliability](https://download.arxiv.org/pdf/2201.13291v3.pdf)**: 

**[ROAR](https://proceedings.neurips.cc/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf)**: 

**[Robustness](https://arxiv.org/pdf/1806.08049.pdf)**: 

**[Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)**: 

**[Stability](https://arxiv.org/pdf/1806.07538.pdf)**: 

**[Sufficiency](https://arxiv.org/pdf/1810.03805.pdf)**: 

## Label Sensitivity
Describe the saliency method's sensitivity to label changes. Provide the results of the saliency method on label sensitivity tests. Known label sensitivity tests are described below:

## Model Sensitivity
Describe the saliency method's sensitivity to model changes. Provide the results of the saliency method on model sensitivity tests. Known model sensitivity tests are described below:

# Perceptibility Testing
Summarize perceptibility considerations of the saliency method that may impact how it is perceived by an end user. 

## Minimality
Describe the saliency method's minimality. Provide the results of the saliency method on minimality tests. Known minimality tests are described blow:

## Perceptual Correspondence
Describe the saliency method's perceptual correspondence. Provide the results of the saliency method on perceptual correspondence tests. Known perceptual correspondence tests are described blow:



# Citation [optional]
If there is a paper or blog post introducing the model, provide the APA and Bibtex information here.

**BibTeX:**

**APA:**
