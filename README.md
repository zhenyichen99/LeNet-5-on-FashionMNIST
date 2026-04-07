# LeNet-5 on FashionMNIST

## Architecture
This project trains the LeNet-5 convolutional network and a dropout varation on the FashionMNIST dataset. The LeNet-5 architecture was published in [Gradient-based learning applied to document recognition](https://hal.science/hal-03926082/document) by LeCun et al. The networks are trained using the [AdamW](https://optimization.cbe.cornell.edu/index.php?title=AdamW) optimizer, with early stopping.

## Files
`model.py` defines the LeNet-5 model and its dropout variant
`train_test.py` contains the training and testing loops, as well as the epoch loop with early stopping
`run.ipynb` is a demo comparing LeNet-5 and the dropout variant

## Findings
Early stopping the training with a patience of 10 epochs, the minimum test loss is usually achieved between epoch 5-10, with test accuracy around 89%. The dropout variation exhibits slightly worse performance (~88%). Some of the hyperparameters were taken from [this paper](https://ieeexplore.ieee.org/document/9047776) and [this paper](https://pdfs.semanticscholar.org/5940/2441f241a01afb3487912d35f75dd7af4c6b.pdf). Both papers claimed a much higher test accuracy (98+%), but I could not replicate these results. I am not sure what my implementation is missing.
