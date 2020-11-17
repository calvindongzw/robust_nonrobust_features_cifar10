# robust_nonrobust_features_cifar10

This is a reproduction work of *"Adversarial Examples Are Not Bugs, They Are Features"* by [Madry's lab](https://github.com/MadryLab/constructed-datasets)

**Models trained on nature CIFAR10**

1. Adversarial Trained Model is trained by L2-PGD Attack.

eps = 0.5

alpha = 0.1

iter = 7

|Model|Clean Accuracy|Adv Accuracy (L2-PGD Attack (eps=0.25))|
|-----|--------------|------------|
|ResNet-20|90.80%|25.70%|
|Adv Trained ResNet-20|83.80%|73.90%|

2. Adversarial Trained Model is trained by Linf-PGD Attack.

eps = 8 / 255

alpha = 2 / 255

iter = 7

|Model|Clean Accuracy|Adv Accuracy (L2-PGD Attack (eps=0.25))|
|-----|--------------|------------|
|ResNet-50|91.27%|3.89%|
|Adv Trained ResNet-50|78.57%|69.53%|

