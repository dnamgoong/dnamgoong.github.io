---
layout: post
title: Knowledge Distillation
date: 2025-09-07
description: study notes
tags: ai transformer
categories: 
related_posts: false
toc:
  beginning: true
---


### Introduction

These are my study notes on knowledge distillation. 

"distillation" is a kind of training that transfers the knowledge from one model to another ([Hinton et al. 2015](https://arxiv.org/abs/1503.02531)).



### Softmax with a Temperature $T$

$$o_j$$ is the $$j$$-th element of the logit vector $${\bf o}$$,
and $$p_j$$ is the corresponding probability from the softmax layer.

* $$ {\bf o} = [o_1, o_2, o_3, \cdots] $$

* $$ {\bf p} = \mbox{softmax}({\bf o}/T) = [p_1, p_2, p_3, \cdots], $$

where  $$T$$ is a temparature that is normally set to 1.

The probability $$p_i$$ is given by

$$ 
p_i  = [\mbox{softmax}({\bf o}/T)]_i = \frac{ \exp\left(\frac{o_i}{T}\right) } { \sum_j \exp\left(\frac{o_j}{T}\right)} , 
$$

where $$[\mbox{softmax}({\bf o}/T)]_i$$ denotes the $$i$$-th element of the vector $$\mbox{softmax}({\bf o}/T)$$.

#### Properties

Assume 

$$o_{i^*} > o_j,$$ 

for $$j \neq i^*$$.

In other words,

$$o_{i^*} = \max_j o_j.$$


The probability $$p_i$$ can be rewritten as

$$ 
p_i 
= \frac{ \exp\left(\frac{o_i}{T}\right) } { \sum_j \exp\left(\frac{o_j}{T}\right)} 
= \frac{ \exp\left(\frac{o_i}{T}\right) } { \sum_j \exp\left(\frac{o_j}{T}\right)} 
\left( \frac{\exp\left(-\frac{o_{i^*}}{T}\right)}{\exp\left(-\frac{o_{i^*}}{T}\right)} \right)
= \frac{ \exp\left(\frac{o_i-o_{i^*}}{T}\right) } { \sum_j \exp\left(\frac{o_j-o_{i^*}}{T}\right)},
$$


$$
p_{i} = \frac{\exp\left(\frac{o_i-o_{i^*}}{T}\right)} { 1 + \sum_{j\neq i^*} \exp\left(\frac{o_j-o_{i^*}}{T}\right)},
$$
 
$$
p_{i^*} = \frac{1} { 1 + \sum_{j\neq i^*} \exp\left(\frac{o_j-o_{i^*}}{T}\right)}
$$


As $$T \rightarrow 0$$, 
* $$\exp\left(\frac{o_j-o_{i^*}}{T}\right) \rightarrow 0,$$ for $$j \neq i^*$$  
* $$p_{i^*} \rightarrow 1$$.
* $$p_i \rightarrow 0,$$ for $$i\neq i^*$$. 
 

As $$T \rightarrow \infty$$, 
* $$\exp\left(\frac{o_j-o_{i^*}}{T}\right) \rightarrow 1,$$ for all $$j$$.
* $$p_i  \rightarrow \frac{1}{N_\text{class}},$$ for all $$j$$.



**Summary**:
* As $$T \rightarrow 0$$, the softmax outputs $$p_i$$ approaches either 0 or 1.
* As $$T \rightarrow \infty$$, the softmax outputs $$p_i$$ become more uniform.

(In large language models like GPT4, the temperature $T>1$ is used to generate more creative responses.) 



### Distilling the knowledge of the teacher into the student

Geoffrey Hinton introduced the idea of knowledge distillation in [[Hinton et al. 2015](https://arxiv.org/abs/1503.02531)], as a way to train a small neural network.


In knowledge distillation, we have two models.
* A **teacher** model -- a big trained neural network. 
    * It is easier to train because "cumbersome models (teacher) can easily extract structure from data." [[Hinton et al. 2015](https://arxiv.org/abs/1503.02531)]
* A **student** model -- a small neural network, which may not be easy to train.

Given an image of a BMW $${\bf x}$$, the teacher's softmax layer output may show that

$$ P(\mbox{ garbage truck } | \ {\bf x} ) \gg P( \mbox{ carrot } | \ {\bf x} ) $$

Hinton argues that probability for incorrect answers tell us a lot about how the teacher tends to generalize. 

We want the student model to generalize in the same way as the teacher model.
For an image classification task, rather than training the student model from scratch using ground-truth labels (e.g., cats, dogs, etc), we train the student model to mimick the softmax outputs from the teacher model, in order to transfer the generalization ability of the teacher model to a student model.



We assume that a trained teacher model is already availble. The teacher model is "frozen"; the teacher model is not updated during the student training.

**Definitions**

* $${\color{red} {\bf y} = [y_1,\ y_2, \cdots, \ y_{N_\text{class}}]} $$ = [one-hot encoding](https://d2l.ai/chapter_linear-classification/softmax-regression.html#subsec-classification-problem) of the correct label.
  * If the image belongs to the class $$k$$, $$y_k = [{\bf y}]_k  = 1$$, and $$y_j=0$$ for $$j\neq k$$. 
  * If the image belongs to the class 1,   $${\bf y} = [1, 0, 0, \cdots, 0, 0]$$.
  * If the image belongs to the class 2,   $${\bf y} = [0, 1, 0, \cdots, 0, 0]$$.
* $${\color{teal}{\bf o}_s} \in \mathbb{R}^{N_\text{class}} $$ = the logits of the student.
* $${\color{red}{\bf o}_t} \in \mathbb{R}^{N_\text{class}} $$ = the logits of the teacher.


#### Supervised training loss

Supervised training of the student will use the [cross-entropy loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). 

For a given image $${\bf x}$$ with the label ${\color{red}{\bf y}}$, the cross-entropy loss is computed by 

$$
\mathcal{L}_{CE} \left(  {\color{teal}\mbox{softmax}({\bf o}_s)}, {\color{red}{\bf y}}  \right)
=
-\sum_{j=1}^{N_\text{class}} {\color{red} y_j} \log [{\color{teal} \mbox{softmax}({\bf o}_s) }]_j.
$$

* $$[{\color{teal} \mbox{softmax}({\bf o}_s)}]_j$$ = the $$j$$-th element of the vector  $$\mbox{softmax}({\bf o}_s)$$
  * $$ [{\color{teal}\mbox{softmax}({\bf o}_s)}]_j = P[\mbox{The image belongs to the class } j \ | \ \mbox{the image}] $$

*  If the image belongs to the class 7,  $$\mathcal{L}_{CE} \left(   {\color{teal} \mbox{softmax}({\bf o}_s) } , {\color{red} {\bf y} }  \right)
= -\log [{\color{teal}\mbox{softmax}({\bf o}_s)}]_7$$ 

By mimizing this loss, we try to maximize the probability of predicting the correct label.
In other words, we try to match the softmax outputs to the one-hot encoding of the correct label $${\color{red} {\bf y}}$$.


#### Distillation loss

The knowledge distillion introduces the cross-entropy with the softmax outputs ("**soft targets**") of the teacher.

$$
\mathcal{L}_{\text{teacher}}
= \mathcal{L}_{CE} \left(
{\color{teal} \mbox{softmax}( {\bf o}_s/T )}, 
\ 
{\color{red} \mbox{softmax}( {\bf o}_t/T )}
\right)
 =
  -\sum_{j=1}^{N_\text{class}} [ {\color{red} \mbox{softmax}({\bf o}_t/T) }]_j \log [ {\color{teal} \mbox{softmax}({\bf o}_s/T) } ]_j ,
$$

In $$\mathcal{L}_{\text{teacher}}$$, the teacher's soft outputs $${\color{red} \mbox{softmax}( {\bf o}_t/T )}$$ plays the same role as the true label $${\color{red} {\bf y}}$$ used in the supervised training.

We train the student on the total loss given by

$$
\mathcal{L}_{KD} = \mathcal{L}_{\text{teacher}} + \lambda \ \mathcal{L}_{CE},
$$

for some $\lambda > 0$.

If $$T=1$$, the relative probabilities of incorrect answers by the teacher may be too small to have any influence on $$\mathcal{L}_{\text{teacher}}$$. Hence, we raise the temperature $$T$$ of the softmax until the teacher model produces a
suitably soft set of targets. 



### Data-efficient image Transformers (DeiT)
 
 References:
* Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou.
Training data-efficient image transformers & distillation through attention. [arXiv:2012.12877](https://arxiv.org/pdf/2012.12877).
* [Machine Learning Street Talk (MLST) podcast, 044](https://www.youtube.com/watch?v=viClVMxiwI0&list=PLwFLAA-F1PgpVziWTKjjomz670oAVEWsp&index=26)


<div class="row mt-4">
    <div class="col-sm-3 mt-3 mt-md-0">
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/knowledge_distillation/DeiT.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-3 mt-3 mt-md-0">
    </div>
</div>

[[Touvron et al. 2021](https://arxiv.org/pdf/2012.12877)] talks about a "data-efficient" stategy to train a [vision transformer](https://dnamgoong.github.io/blog/2025/vision-transformer/); "data efficieny" or "sample efficieny" means that smaller amount of training data is needed to achieve good performance.



**Definitions**

* $$y$$ = the ground truth.
* $$Z_s^{\text{CLASS}}$$ = = the logits of the student model from the class token **[class]**.
* $$Z_s$$ = the logits of the student model from the distillation token **[distill]**.
* $$Z_t$$ = the logits of the teacher model.


**Summary**
* Teacher = a ConvNet
* Training objective is a linear combination of
  * The supervised training loss: $$\mathcal{L}_{CE} \left( \mbox{softmax}(Z_s^{\mbox{CLS}}), \ y \right)$$
  * The distillation loss:  $$\mathcal{L}_{\mbox{teacher}}$$


##### Soft distillation
*  
  
  $$\mathcal{L}_{global} = (1-\lambda) \mathcal{L}_{CE} \left( \mbox{softmax}(Z_s^{\mbox{CLS}}), \ y \right)
   + \lambda T^2 \mathcal{L}_{\mbox{teacher}} $$

*  
  
  $$ \mathcal{L}_{\mbox{teacher}} = KL \left(\mbox{softmax}(Z_t/T) \ \Vert \ \mbox{softmax}(Z_s/T) \right)$$