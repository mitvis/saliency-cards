# Saliency Cards
![Saliency card teaser.](teaser.png)

**Saliency Cards** are transparency documentation for saliency methods. This repo contains a saliency card template and saliency cards for existing saliency methods. Use saliency cards to analyze and compare saliency methods based on the attributes important to your tasks!

*We accept pull requests!*

Saliency cards are living artifacts that evolve to support the rapid growth of saliency method research. This repository is designed as a centralized location for saliency method documentation. As new saliency methods and evaluation techniques are developed, saliency cards will need to be added and uploaded.

## Contributing New Saliency Methods
If you have developed a new saliency method or want to contribute a saliency card for an existing saliency method, please use the `saliency_card_template.md` to create a saliency card by filling out each appropriate section. Name the saliency card `{method name]}_saliency_card.md` and store in its own folder in the root of this directory.

## Contributing New Evaluations
If you have new evaluation metrics or the results of existing evaluation metrics on untested saliency methods, add the metric and a short summary to the appropriate section of `saliency_card_template.md`. Add the evaluation results to the saliency cards of any methods you evaluated.

## Citation
```
@inproceedings{saliencycards,
  title={{Saliency Cards: A Framework to Characterize and Compare Saliency Methods}},
  author={Boggust, Angie and Suresh, Harini and Strobelt, Hendrik and Guttag, John V and Satyanarayan, Arvind},
  booktitle = {ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
  year={2023}
}
```
