# Saliency Methods

In the saliency card attribute examples, we use a variety of saliency methods. To have a consistent interface across their public implementations, we provide a wrapper.

This directory contains implementations of common saliency methods. See the [example notebook](https://github.com/mitvis/saliency-cards/blob/main/attribute_examples/saliency_method_examples.ipynb)!

Implemented saliency methods:
* **Vanilla Gradients** ([paper 1](https://www.researchgate.net/profile/Aaron_Courville/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network/links/53ff82b00cf24c81027da530.pdf) | [paper 2](https://arxiv.org/pdf/1312.6034.pdf) | implemented via [Captum](https://captum.ai/api/saliency.html))
* **Input x Gradient** ([paper](https://arxiv.org/pdf/1605.01713.pdf) | implemented via [Captum](https://captum.ai/api/input_x_gradient.html))
* **Integrated Gradients** ([paper](https://arxiv.org/pdf/1703.01365.pdf) | implemented via [Captum](https://captum.ai/api/integrated_gradients.html))
* **SmoothGrad** ([paper](https://arxiv.org/abs/1706.03825.pdf) | adapted from [Google Pair Saliency](https://github.com/PAIR-code/saliency))
* **Guided Backprop** ([paper](https://arxiv.org/pdf/1412.6806.pdf) | implemented via [Captum](https://captum.ai/api/guided_backprop.html))
* **GradCAM** ([paper](https://arxiv.org/pdf/1610.02391.pdf) | implemented via [Captum](https://captum.ai/api/layer.html#gradcam))
* **Gradient SHAP** ([paper](https://arxiv.org/pdf/1705.07874.pdf) | implemented via [Captum](https://captum.ai/api/gradient_shap.html))
* **Kernel SHAP** ([paper](https://arxiv.org/pdf/1705.07874.pdf) | implemented via [Captum](https://captum.ai/api/kernel_shap.html))
* **RISE** ([paper](https://arxiv.org/pdf/1806.07421.pdf) | implemented via [the authors' GitHub](https://github.com/eclique/RISE))
* **XRAI** ([paper](https://arxiv.org/pdf/1906.02825.pdf) | adapted from [Google Pair Saliency](https://github.com/PAIR-code/saliency))
* **LIME** ([paper](https://arxiv.org/pdf/1602.04938.pdf) | implemented via the [authors' GitHub](https://github.com/marcotcr/lime))
* **SIS** ([paper 1](https://arxiv.org/pdf/1810.03805.pdf) | [paper 2](https://arxiv.org/pdf/2003.08907.pdf) | implemented via the [authors' GitHub](https://github.com/gifford-lab/overinterpretation))

## Usage
Each saliency method (i.e., `VanillaGradients`) extends the base class `SaliencyMethod`. Each method is instantiated with a model and, optionally, other method specific parameters. An `SaliencyMethod` object has two public methods: `get_saliency` and `get_saliency_smoothed`. 

`get_saliency` takes in an `input_batch` (e.g., a batch of images) and outputs an `np.array` of the same size that represents the attributions. It defaults to computing the saliency with respect the the model's predicted class, but `target_classes` can optionally be passed to specify a specific class. `target_classes` is a list of integers the same length as the batch size. `target_class[i]` is the index of the class to compute saliency with respect to for `input_batch[i]`.

`get_saliency_smoothed` applies SmoothGrad to the `get_saliency` attributions.

Once saliency is computed, [`util.py`](https://github.mit.edu/aboggust/interpretability_methods/blob/master/util.py) contains code to visualize the attributions.

### Example:
```
# Getting Vanilla Gradients with respect to the predicted class.
from vanilla_gradients import VanillaGradients
from util import visualize_saliency

model = ... # assuming pytorch model 
input_batch = ... # assuming 4D input batch (batch, channels, height, width)
vg = VanillaGradients(model)
vg_saliency = vg(input_batch) # attributions of shape (batch, channels, height, width)
visualize_saliency(vg_saliency) # will output grayscale saliency image
```

