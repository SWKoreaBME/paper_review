# What uncertainties Do we need in bayesian deep learning for computer vision?



## Abstract

1. Two major types of uncertainty
   1. Aleatoric - U in observation
   2. Epistemic - U in model
2. Traditionally difficulty in modeling epistemic uncertainty
3. Contributions
   1. Study **benefits of modeling epistemic vs. aleatoric U in BDL for vision tasks** => Present BDL framework combining input-dependent aleatoric U together w/ epistemic U
   2. Study models with segmentation and depth regression tasks
   3. New Loss fuction => Interpreted as learned attenuation
   4. **Makes the loss more robust to noisy data**



## Introduction



## Conclusions

1. BDL framework to learn a mapping to aleatoric uncertainty from the input data, **which is composed on top of epistemic U models.**

2. Alea U
   1. Large data situations
   2. Real-time applications
3. Epist U
   1. Safety-critical applications
   2. Small datasets
4. **E-U and A-U is not mutually exclusive** => Showed that the combination can achieve new SOTA results