<!-- Adapted from the Huggingface Model Card template: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md -->
---
For more information on saliency cards, see: [Saliency Cards: A Framework to Characterize and Compare Saliency Methods](https://arxiv.org/abs/2206.02958)

# Saliency Card for {method name}
Provide a quick summary of the saliency method.

# Methodology
Provide a longer summary of what this saliency method does, its intended use, and important considerations.
- **Developed by:** {developers}
- **Shared by [optional]:** {who is sharing it}
- **References:** {reference to the paper, demos, and/or source code}
- **Example:** {show a visual example of the method}

## Determinism
Describe the saliency method's sources of non-determinism.

## Hyperparameter Dependence
Describe the saliency method's hyperparameters and provide suggestions on how to set them.

## Model Agnosticism
Describe the saliency method's dependence on specific models --- for instance, expectations of architecture types of differentiability.

## Computational Efficiency
Describe the saliency method's computational efficiency and computing expectations.

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

# Citation [optional]
If there is a paper or blog post introducing the model, provide the APA and Bibtex information here.

**BibTeX:**

**APA:**
