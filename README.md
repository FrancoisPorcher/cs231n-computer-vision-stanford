# C231n - Stanford - Computer Vision

<p align="center">
 <img src="img/cover.png" width="60%" height="60%">
</p> 

## Content of this class
This is a (non-exhaustive) summary of the content of the CS231N class (2023).

- SVM, Softmax, Regularization
- CNN
- Generative Models, GANs
- Transformers, Attention
- Hardware and Algorithmics for Modern Deep Learning
- Reinforcement Learning
- Self-Supervised Learning


These are my assignments and lecture notes from Fei-Fei Li's famous Computer Vision Class. (I will upload the assignments soon).

<p align="center">
 <img src="CheatSheet CS231N.png" width="60%" height="60%">
</p> 


## Contents
- [Highlights](#highlights)
 - [Image Descriptions with transformers](#image-descriptions-with-transformers)
 - [Saliency Map](#saliency-map)
- [Fooling a Neural Network](#fooling-a-neural-network)
- [Deep Dream](#deep-dream)
- [Official Stanford Git Repo](#official-stanford-git-repo)



## Highlights

Here are the highlights of this class for me:

### Image Descriptions with transformers.



Cool way to map images to captions with transformers. Train on the [Coco](https://cocodataset.org/#home) dataset.

<p align="center">
 <img src="img/image_description.png" width="60%" height="60%">
</p> 

### Saliency Map

We compute the derivative of the entry pixels in regards to the target class.
I liked this method but GradCam is more powerful in my opinion.

<p align="center">
 <img src="img/saliency.png" width="60%" height="60%">
</p> 

### Fooling a Neural Network

Starting from an image and moving in the direction of the gradient, we can actually fool a neural network. The interesting thing is that to the human eye the images are the same. But in fact every pixel is chery picked to activate the neurons in a certain way to predict the wrong output.

<p align="center">
 <img src="img/gradient_ascent.png" width="60%" height="60%">
</p> 

### Deep Dream

Starting from a noise image, we move in the gradient direction to output a specific class. This creates very dreamy images. We can actually visualize some bottles across the different epochs.

<p align="center">
 <img src="img/deepdream2.png" width="60%" height="60%">
</p> 


## Official Stanford Git Repo

[Stanford Official Git Repo](https://github.com/cs231n/cs231n.github.io)

