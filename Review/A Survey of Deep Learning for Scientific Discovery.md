# A Survey of Deep Learning for Scientific Discovery

by Maithra Raghu, Eric Schmidt @ Google

> In this survey, we focus on addressing this central issue, **providing an overview of many widely used deep learning models, spanning visual, sequential and graph structured data, associated tasks and different training methods, along with techniques to use deep learning with less data and better interpret these complex models** — two central considerations for many scientific use cases
>
> We also **include overviews of the full design process**, implementation tips, and links to a plethora of tutorials, research summaries and open-sourced deep learning pipelines and pretrained models, developed by the community

------

## 7 Interpretability, Model Inspection and Representation Analysis

> **Identifying underlying mechanisms** giving rise to **observed patterns in the data**. When **applying deep learning in scientific settings**, we can use these observed phenomena as prediction targets, but the **ultimate goal** remains **to understand what attributes give rise to these observations**

1. In the **Scientific field**, **Deep Learning** is often used to **UNDERSTAND** **a certain phenomenon**
   - For example
     - Input : **Amino acid**, output : **the predicted properties of the protein**
     - Understanding how that amino acid sequence resulted in the observed protein function
   - Interpretability techniques
     - **Fully understandable**, **step-by-step explanation** of the model's decision process
     - From **feature attributions** to **model inspection**
       - feature attr : Determining **what input features matter** the most
       - model ins. : Determining **what causes** neurons in the network to fire
       - These two types provide a rough split in the type of interpretability
2. **Two Distinctive methods**
   1. **Feature Attribution** ( Per Example Interpretability)
      1. Concentrates on **taking a specific input** along **with a trained deep neural network**
      2. Determining **what features of the input** are most important
   2. **Model Inspection** ( Model Inspection and Representational Analysis )
      1. **Revealing important, hidden patterns in the data** that the **model has implicitly learned** through being trained on the predictive task
      2. Example, **Machine translation** => **representation analysis techniques** used to illustrate latent linguistic structure learned by the model

## 7.1 Feature Attribution and Per Example Interpretability

Feature attr. @ a per example level => **Answering questions** such as **which parts of an input image** are **most important** for a particular model prediction

### 7.1.1 Saliency Maps and Input Masks

1. Saliency maps
   - Take the **Gradient of the output prediction with respect to the input**
   - Gives **mask** over the input => Highlighting **which regions have large gradients** ( most important for the prediction )
2. Example => These **inspections are not perfect** though
   - Grad-CAM
   - SmoothGrad
   - IntGrad
   - https://github.com/PAIR-code/saliency
   - " The building blocks of interpretability " 
     - Provides **the ability** to inspect **the kinds of features causing neurons across diff. hidden layers to fire**
     - https://distill.pub/2018/building-blocks/
     - https://github.com/tensorflow/lucid
   - Using Deconvolutional layers

### 7.1.2 Feature Ablations and Perturbations

**Isolate the crucial features of the input** either **by performing *feature ablations*** or **computing *perturbations* of the input** and **using these perturbations** along with the original input **to inform** the **importance of different features**.

1. Feature Ablations
   1. The notion of ***Shapely  value***
      1. Estimates **the importance of a particular feature x_0** in the input **by computing the predictive power of a subset of input features**
      2. And averaging over all possible subsets
2. Utilizing perturbations
   1. LIME
      - Uses **multiple local perturbations** to enable learning an interpretable local model
   2. DEEPLIFT
      - Uses a reference input to compare activation differences

## 7.2 Model Inspection and Representation Analysis

Gaining insight not at a single input example level, but **using a set of examples to understand the salient properties of the data**

### 7.2.1 Probing and Activating Hidden Neurons

**(i) Understanding what kinds of inputs it activates for**

**(ii) Directly optimizing the input to activate a hidden neuron**

1. Network Dissection
   - **Hidden neurons** are categorized by **the kinds of features** they respond to
   - http://netdissect.csail.mit.edu/
2. Take a **NN**, **fix its params** and **optimize the input** to find the kinds of features that makes some hidden neuron activate

### 7.2.2 Dimensionality Reduction on Neural Network Hidden Representations

1. Dimensionality Reduction in standard scientific settings

   Useful in **revealing important factors of variation** and **critical differences in the data subpopulations**

   1. PCA ( Principal component Analysis )
   2. t-SNE
   3. UMAP

2. The **NN may implicitly learn these important data attrs**. in its hidden representations **which can then be extracted through dimensionality reduction methods**.

### 7.2.3 Representational Comparisons and Similarity

1. A line of work has studied **comparing hidden representations across different NN models**
   1. Matching algorithms
   2. Canonical Correlation analysis
      1. Used to identify and understand many representational properties in NLP applications
      2. Modelling the mouse visual cortex as an ANN
   3. **Kernel based approach** to perform **similarity comparisons**
      1. "Similarity of Neural Network Representations Revisited" with code implementations



------

## 8 Advanced Deep Learning Methods

- We term these methods 'Advanced' as they are more intricate to implement and may require specific properties of the problem to be useful

## 8.1 Generative Models

**Generative modelling has two fundamental goals**

1. Seeks to **model and enable sampling** from high dimensional data distributions, such as natural images
   1. **Generative models** that take samples of the high dim. distribution as input and **learn some task directly on these data instances**
   2. If generative modelling achieved perfect success at this first goal => Possible to continuously sample 'free' data instances
2. Looks to learn **lower dimensional latent encodings** of the data **that capture key properties of interest**
   1. **Learning latent encodings** of the data with **different encoding dimensions correspond to meaningful factors of variation** ( i.e. VAE )

### 8.1.1 Generative Adversarial Networks

1. GAN
   1. Consist of two neural networks
      1. Generator 
         1. Takes as **input a random noise vector** and tries to output samples that look like the data distribution
      2. Discriminator
         1. Tries to distinguish between true samples of the data, and those synthesized by the generator

##### Unconditional GANs vs Conditional GANs

Examples above  are all unconditional GANs, where the data is generated with only a random noise vector as input

> Popular and highly useful variant are conditional GANs, where generation is conditioned on additional information, such as a label, or as 'source' image, which might be translated to a different style

For example, Pix2pix, cycleGAN.

### 8.1.2 Variational Autoencoders (VAE)

VAEs have an **encoder decoder structure** => Explicit latent encoding which can capture useful properties of the data distribution

Also **enable estimation of the likelihood of a sampled datapoint** - The prob. of its occurence in the data distribution.

1. Many VAEs Popularity
   1. Because of **the explicit latent encoding** and **the ability to estimate likelihoods**
   2. Examples
      1. modelling gene expression in single-cell RNA sequencing

### 8.1.3 Autotregressive Models

1. Autoregressive models
   - Take in **inputs sequentially** and **use those to generate an appropriate output** and use these to generate a new pixel value for a specific spatial location
   - Examples
     - PixelRNN
     - VQ-VAE

### 8.1.4 Flow Models

1. **Flow model** looks at **performing generation using a sequence of invertible transformations** = enables the computation of exact likelihoods

   - Perfoming an expressive and tractable sequence of invertible likelihoods

   - Video tutorial : https://www.youtube.com/watch?v=i7LjDvsLWCg&feature=youtu.be

## 8.2 Reinforcement Learning

- RL aims to solve the sequential decision making problem
- Introduced with the notions of environment and an agent.
- The goal of RL is to learn
  - through interaction with the env.
  - good sequences of actions ( a.k.a policy )
- Unlike supervised learning, feedback ( reward ) is typically given only after performing the entire sequence of actions

### 8.2.1 RL with an Environment Model/Simulator

1. **A variety of learning algs**. can help **the agent** learn a good sequence of actions often through **simultaneously learning a value function** ( AlphaGoZero )
   - Value function : A function that determines whether a particular env. state is beneficial or not
   - Very important in **properly assessing the value of the environment state**
   - Combining value functions with traditional search algorithms has been a very powerful way to use RL

### 8.2.2 RL without Simulators

1. Don't have access to an environment model/simulator just have records of sequences of actions = **The offline settings**
   - Trying to teach an agent a good policy but thorough validation and evaluation can be challenging
2. Evaluation in **off-policy settings** often uses a statistical technique known as off-policy policy evaluation
   - In robotics, RL literature has looked at performing **transfer learning btw policies learned in simulation and policies learned on real data**