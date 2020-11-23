# Robust Features

Zhiwei Dong

Duke University 

zhiwei.dong@duke.edu

**Abstract**

## Introduction

Since Szegedy et al. (2014b) found the vulunerability of Deep Neural Network (DNN), adversarial attacks and adversarial training have become two hot topics in deep learning field. Goodfellow et al. (2015) developed the Fast Gradient Sign Method (FGSM) which can easily fool an avdanced DNN architecture with tiny perturbations that even couldn't be diffrentiate by human eyes [1]. Carlini and Wagner (2017) created Carlini-Wagner (CW) loss function that significantly improved the attack rate [2] and Madry et al. (2019) proposed Projected Gradient Decent (PGD) [3] that has been using pervasively in adversartial attack/training researches. 

Madry et al. (2019) found the reason why state-of-the-art DNN architectures are vulnerable is that the are prone to learn from non-robust features in raw images. They explored CIFAR-10 and found features are not in same robustness level. Some of them are robust features which means information learned from them are robust against with attacking. While, some features are not robust. They are vulnerable when attacking by adversarial examples. In this project, I reproduced robust CIFAR-10 generation process and tested whether the DNN models trained by robust features is more robust than the DNN model trained by original CIFAR-10 dataset.

## Methods

To create a robust CIFAR-10 dataset, we need to only extract robust features and remove non-robust features, which means we actually change the distribution of variables in the original dataset. Therefore, the challange is how to find a distribution of robust features in the original CIFAR-10. An intuitive approach is to develop a robust DNN model based on the original CIFAR-10 because adversarial trained models have been demonstrated suffient robustness in most of popular adversarial attacking settings [3]. Since robust DNN models extract effetive and robust features through convolutinal layers and use activations of the final convolutional layer as inputs of fully-connected layers, the representation layer of a robust DNN model saves the information of distributions of robust features. Therefore, the pipeline of this project is: 1) creating a baseline DNN model trained by the original CIFAR-10 dataset and a robust DNN model adversarilly trained by PGD attacked examples; 2) finding a mapping function to closely match each original example to a robust exmaple; 3) training a new DNN model by the robust CIFAR-10 dataset; 4) testing the new DNN model on adversarial examples.

### Baseline Model



