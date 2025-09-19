---
layout: post
title: Linear Self-Attention for Regression
date: 2025-08-30
description: Regression
tags: code math ai transformer
categories: 
related_posts: false
toc:
  beginning: true
---




## Introduction

In this post, we implement the linear attention layer based regression described in [[Lu et al. 2024](https://arxiv.org/abs/2405.11751)]
* Yue M. Lu, Mary I. Letey, Jacob A. Zavatone-Veth, Anindita Maiti, and Cengiz Pehlevan, Asymptotic theory of in-context learning by linear attention.
  * Blog: Mary Letey, [Solvable Model of In-Context Learning Using Linear Attention](https://kempnerinstitute.harvard.edu/research/deeper-learning/solvable-model-of-in-context-learning-using-linear-attention/), July 2025.

(Ref: Gregory Valiant's talk in [Large Language Models and Transformers Workshop](https://simons.berkeley.edu/talks/gregory-valiant-stanford-university-2023-08-18) at Simon Institute.)



[[Raventos et al. 2023](https://arxiv.org/abs/2306.15063)] uses a transformer model to study the emergence of in-context learning for regression, whereas [[Lu et al. 2023](https://arxiv.org/abs/2405.11751)] uses a simple linear self-attention layer.


To be consistent with [my previous post](https://dnamgoong.github.io/blog/2025/ridge/) on ridge regression, we will use the same notations from [[Raventos et al. 2023](https://arxiv.org/abs/2306.15063)].



## Data distribution

For each batch element, we generate the data as follows

$$ 
y = {\bf{w}}^T {\bf{x}} + \epsilon = \sum_{i=0}^{D-1} w_i x_i + \epsilon
$$


* $${\bf{w}} \in \mathbb{R}^D$$, $${\bf{x}} \in \mathbb{R}^D$$.
* $${\bf{x}} \sim \mathcal{N}({\bf 0}, \frac{1}{\sqrt{D}}{\bf{I}}_D)$$, which means $$E[\bf{x}]=0$$ and $$E[{\bf xx}^T]= \frac{1}{D}{\bf I}_D$$.
* "Task vector": $${\bf w} \sim \mathcal{T}_{true} = \mathcal{N}({\bf 0}, {\bf I}_D)$$, and the test data is generated according to this model.
* observation noise: $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$, which means $$E[\epsilon]=0$$, and $$E[\epsilon^2]= \sigma^2$$
* $K$ = the number of examples $$({\bf x}, y)$$ generated for the prompt in each batch element.
  * The last one will become a query.

But, for a pre-training dataset, we sample {$${\bf w}^{(1)}, {\bf w}^{(2)}, \cdots, {\bf w}^{(M)} $$} once from $$\mathcal{T}_{true}$$
* Each batch element is generated using one of the $$M$$ tasks {$${\bf w}^{(1)}, {\bf w}^{(2)}, \cdots, {\bf w}^{(M)}$$}.


In [[Raventos et al. 2023](https://arxiv.org/abs/2306.15063)], the distrubtion of $${\bf w}$$ in the pretraining data is denoted by 

$$
\mathcal{T}_{pretrain} = \mathcal{U} ({\bf w}^{(1)}, {\bf w}^{(2)}, \cdots, {\bf w}^{(M)} )
$$

For each batch element, a "prompt" consists of a bunch of $$K-1$$ examples $$({\bf x}, y)$$ generated from one $$\bf{w}$$, and a query $$\bf{x}_{query}$$. 

Given a prompt that consists of $$K-1$$ examples ($$({\bf x}_i, y_i)$$, for $$i=1,2,\cdots, K-1$$), 

we want to predict $$y$$, corresponding to the query $${\bf x}_{query} = {\bf x}_K$$. How to do the prediction is explained in the next section.

To quantify the accuracy of the predictions, the MSE is used; the average squared difference between the true $$y_K$$ and its prediction $${\hat y}_K$$. 





## A Linear Self-Attention Layer for Regression 

We will provide a summary of the in-context learning model used in [[Lu et al. 2024](https://arxiv.org/abs/2405.11751)]. 

The embedding of the prompt including the query is represented by a $$(D+1) \times K$$ matrix $$Z$$,

$$ 
Z = \left[
\begin{matrix}
{\bf x}_1 & {\bf x}_2 & \cdots & {\bf x}_{K-1} & {\bf x}_K \\
y_1       &  y_2      & \cdots & y_{K-1}       & 0 
\end{matrix}
\right]
$$

Each column of $$Z$$ is called a token.


$$H_Z$$ is a $$D \times (D+1)$$ matrix defined for each prompt (See Eq (11) of [[Lu et al. 2024](https://arxiv.org/abs/2405.11751)]),

$$
H_Z = {\bf x}_{K} \left[ \frac{D}{K-1} \sum_{i=1}^{K-1} y_i {\bf x}_i^T, \ \ \frac{1}{K-1} \sum_{i=1}^{K-1} y_i^2 \right],
$$

where 
$${\bf x}_K = {\bf x}_{query}$$


$\Gamma$ is a $$D \times (D+1)$$ matrix of the linear self-attention layer parameters to be optmized to minimize the MSE.


The prediction of $$y_K$$ based on the linear self-attention layer outputs can be approximated by

$$
 \hat{y}_K = \ < \Gamma, H_Z > \ = \mbox{Trace}( H_Z^T \Gamma ) 
$$

where $$<\cdot, \cdot>$$ operator computes the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product), 
and [Trace operator](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) computes the sum of the diagonal elements of $$H_Z^T\Gamma$$. One could learn the paramter matrix $\Gamma$ that minimizes the MSE, by defining it as learnable parameters, using [`nn.Parameters`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) in PyTorch. 
For example, 
```python
  self.Gamma = nn.Parameter(torch.randn(D, D+1))
```



But, the optimal solution can be derived as shown in the paper.
Without the ridge regularization (i.e. $$\lambda=0$$), the optimum parameter matrix $$\Gamma^*$$ is computed using the pretraining data (See Eq. (15) of [Lu et al.](https://arxiv.org/abs/2405.11751))

$$
\mbox{vec}(\Gamma^*) = \left( \sum_{\mu=1}^n \mbox{vec}(H_{Z^{\mu}}) \ \mbox{vec}(H_{Z^{\mu}})^T \right)^{-1} \left(\sum_{\mu=1}^n y_K^\mu \ \mbox{vec}(H_{Z^{\mu}}) \right)
$$

* $$\mu$$ is a prompt index
* $$n$$ is the number of the prompts in the pretraining dataset.
* Each prompt $$\mu$$ has $$K-1$$ examples and one query.
* $$\mbox{vec}(A)$$ is a vectorization operation, where a vector is formed by stacking the rows of the matrix $A$.

A MATLAB implementation of the $$\Gamma$$ can be found in [my github repo](https://github.com/dnamgoong/linear-attention-for-regression/blob/main/pretrain_Gamma.m).


```matlab

function [vGamma, Gamma] = pretrain_Gamma(D, M, K, sigma_sq, W_pretrain, pretrain_flag)

% M tasks
% W_pretrain = randn(D, M); % each column contains a task vector w

A=0;
B=0;

num_iterations = 1e6;
for n=1:num_iterations

  if pretrain_flag
    w = W_pretrain(:, mod(n, M)+1);
  else
    w = randn(D,1);
  end

  X = randn(K,D)/sqrt(D);
  epsilon = sqrt(sigma_sq)*randn(K,1);
  y = X*w + epsilon;

  X_context = X(1:K-1, :); % K-1, D
  y_context = y(1:K-1, :); % K-1

  x_query = X(K, :);
  Hz = x_query'/(K-1)*[D*y_context'*X_context, sum(y_context.^2)]; % D, D+1

  vHz = Hz';
  vHz = vHz(:); % stack the rows of Hz

  A = A + vHz*vHz';
  B = B + y(K)*vHz;

end

lamda=0; % no ridge
vGamma = pinv(num_iterations/D*lamda*eye(length(vHz)) + A )*B;
Gamma = reshape(vGamma, D+1, D)';


end
```

#### Structure of the parameter matrix $$\Gamma^*$$

For the task dimension $$D=8$$, we vary the task diversity $$M$$, and the context length $$K$$ to study the structure of $$\Gamma^*$$. We use a [MATLAB script](https://github.com/dnamgoong/linear-attention-for-regression/blob/main/visualize_Gamma.m) to visualize the absolute values of the entries in the matrix $$\Gamma$$ for each $$M$$ and $$K$$.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/linear_attention_regression/thermal_Gamma_D8_K16_M4.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/linear_attention_regression/thermal_Gamma_D8_K16_M16.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/linear_attention_regression/thermal_Gamma_D8_K16_M128.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/linear_attention_regression/thermal_Gamma_D8_K16_M512.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Empirically, we observe that if task diversity $$M$$ is large, the matrix $$\Gamma^*$$ that minimizes the MSE has the following characteristics.

* `Gamma(:, 1:D)` is nearly diagonal and proportional to $${\bf I}_D$$.
* `Gamma(:, D+1)` is nearly a zero vector $${\bf 0}$$.

Hence, for sufficiently large task diversity $$M$$, the matrix $$\Gamma^*$$ may be approximated by 

$$\Gamma^* \approx \alpha\  [{\bf I}_D, \ {\bf 0}_{D \times 1}],$$ 

for some constant $$\alpha < 1$$.


Define a $$(K-1) \times D$$ matrix

$$ 
X = \left[ 
  \begin{matrix} 
  && {\bf x}_1^T &&\\
  && {\bf x}_2^T &&\\
  &&  \vdots &&\\
  && {\bf x}_{K-1}^T &&
  \end{matrix}
\right] 
$$


and a $$(K-1) \times 1$$ vector 

$$ 
{\bf y} = \left[ 
  \begin{matrix} 
  y_1 \\
  y_2 \\
  \vdots \\
  y_{K-1}
  \end{matrix}
\right] 
$$


Using $$X$$ and $${\bf y}$$, the feature matrix $$H_Z$$ can be expressed as

$$
H_Z = {\bf x}_{K} \left[ \frac{D}{K-1} {\bf y}^T X, \ \ \frac{1}{K-1} \sum_{i=1}^{K-1} y_i^2 \right]
$$



Then, the prediction can be approximated as 

$$ 
\hat{y}_{K} =  \mbox{Trace}(\Gamma^* H_Z^T) 
\approx \mbox{Trace} \left(
  \alpha\  [{\bf I}_D, \ {\bf 0}_{D \times 1}] 
\left[ 
  \begin{matrix}
  \frac{D}{K-1} X^T {\bf y}\\
  \frac{1}{K-1} \sum_{i=1}^{K-1} y_i^2 
  \end{matrix}
  \right]   {\bf x}_K^T         
  \right)
$$

(We used $$(AB)^T = B^T A^T$$.)

$$ \hat{y}_K \approx 
\mbox{Trace} \left(
  \alpha \
\frac{D}{K-1}  X^T {\bf y}    {\bf x}_K^T         
  \right) 
$$

Using the property $$Trace(AB) = Trace(BA)$$, 
we have

$$
\hat{y}_K \approx    \mbox{Trace} \left( 
  \alpha \frac{D}{K-1}    {\bf x}_K^T   X^T {\bf y}  \right) ,     
$$

for some constant $$\alpha < 1$$. 

Note that we can remove Trace because $${\bf x}_K^T   X^T {\bf y}$$ is a scalar. Therefore,

$$
\hat{y}_K \approx      
  \alpha \frac{D}{K-1}    {\bf x}_K^T   X^T {\bf y}   ,     
$$

for some constant $$\alpha < 1$$. 


Recall that the [ridge regression estimator](https://dnamgoong.github.io/blog/2025/ridge/) is given by

$$
w_{ridge} = (X^T X + \sigma^2 I )^{-1} X^T {\bf y}
$$

and its prediction by

$$
\hat{y}_{K, \ ridge} = {\bf x}_K^T w_{ridge} = {\bf x}_K^T (X^T X + \sigma^2 I )^{-1} X^T {\bf y}
$$

Note that

$$
\hat{y}_K \neq \hat{y}_{K, \ ridge}
$$


Hence, the MSE of $$\Gamma^*$$ evaluated on $$\mathcal{T}_{true}$$ will be worse than that of the ridge regression estimator $$w_{ridge}$$.

