# Robust Features

Zhiwei Dong

Duke University 

zhiwei.dong@duke.edu

**Abstract**

## Introduction

Since Szegedy et al. (2014b) found the vulunerability of Deep Neural Network (DNN), adversarial attacks and adversarial training has become two hot topics in deep learning field. Goodfellow et al. (2015) developed the Fast Gradient Sign Method (FGSM) which can easily fool an avdanced DNN architecture with tiny perturbation that even couldn't be diffrentiate by human eyes. Carlini and Wagner (2017) created Carlini-Wagner (CW) loss function and Madry et al. (2019) proposed Projected Gradient Decent (PGD) that has been using pervasively in adversartial attack/training researches. 

Madry et al. (2019) found the reason why state-of-the-art DNN architectures are vulnerable is that the are prone to learn from non-robust features in raw images. They explored CIFAR-10 and found features are not in same robustness level. Some of them are robust features which means information learned from them are robust against with attacking. While, some features are not robust. They are vulnerable when attacking by adversarial examples. In this project, I reproduced robust CIFAR-10 generation process and tested whether the DNN models trained by robust features is more robust than the baseline model.

