---
layout: post
title: Vision Transformer
date: 2025-07-01
description: ViT notes
tags: ai transformer
categories: 
related_posts: false
---



 These are my study notes on [Vision Transformer](https://arxiv.org/abs/2010.11929):
* Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, 
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, 
Jakob Uszkoreit, Neil Houlsby. 
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
  


Very nice visualization of Fig.1 of [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929) by [Phil Wang (lucidrains)](https://github.com/lucidrains/vit-pytorch/blob/main/images/vit.gif):

<center width="100%"><img src="/assets/img/vit/vit.gif" width="500px"></center>



### Patchification

* An image ${\bf x}$ of shape `(H, W, 3)` is split into fixed sized patches, where each patch has shape `(P, P, 3)`.  
  * We have 9 patches in the figure.
  * $P$ can be 16 for "16x16 words".

* Each patch is flattened to a row vector ${\bf x}_p^i$ of length $3P^2$, for $i=1,2, \cdots, 9$.
* Linearly embed each of ${\bf x}_p^i$, for $i=1,2, \cdots, 9$, i.e.
  $\textcolor{red}{\bf x}_p^i \textcolor{red}{\bf E}$.
  * $\textcolor{red}{\bf E}$ is a $3P^2 \times D$ matrix.
* Prepend a learnable embedding ${\bf x}_{class}$ ("classification token") to the sequence of embedded patches to obtain
a $10 \times D$ matrix

 $$
 \left[
\begin{matrix}
\textcolor{red}{\bf x}_{class} \\
\textcolor{red}{\bf x}_p^1  \textcolor{red}{\bf E} \\
\textcolor{red}{\bf x}_p^2  \textcolor{red}{\bf E} \\
\vdots \\
\textcolor{red}{\bf x}_p^9  \textcolor{red}{\bf E} \\
\end{matrix}
\right]
 $$   

* Add position embeddings ${\bf E}_{pos}$ ( Eq (1) in [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929) ):

$$
{\bf z}_0 = 
 \left[
\begin{matrix}
\textcolor{red}{\bf x}_{class} \\
\textcolor{red}{\bf x}_p^1  \textcolor{red}{\bf E} \\
\textcolor{red}{\bf x}_p^2  \textcolor{red}{\bf E} \\
\vdots \\
\textcolor{red}{\bf x}_p^9  \textcolor{red}{\bf E} \\
\end{matrix}
\right]
+
\textcolor{purple}{\bf E}_{pos} \ \in \mathbb{R}^{10 \times D}
$$   


 The learnable embedding and the position embedding 
 are implemented in PyTorch using ``nn.Parameter``.

 [[Phill Wang's implementation](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)]:
 
 ```python
# x_class
self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
# E_pos
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
 ```

The 10 row vectors or ``embedded patches'' in ${\bf z}_0$ are fed to a standard transformer encoder, which we will describe next.



### Transformer Encoder
A transformer encoder is formed by stacking $L$ transformer encoder layers. 

<center width="100%"><img src="/assets/img/vit/transformer_encoder.png" width="200px"></center>




Suppose the $l$-th layer is denoted by a function $f_{\theta^{(l)}}$, where $\theta^{(l)}$ represents the neural network paramters of the $l$-th layer.

The 10 embedded patches in ${\bf z}_0$ are fed to 
the first transformer encoder layer 
$$f_{\theta^{(1)}}$$. 

$$ {\bf z}_1 = f_{\theta^{(1)}} ({\bf z}_0) $$

$$ {\bf z}_2 = f_{\theta^{(2)}} ({\bf z}_1) $$

$$ \ \ \ \ \ \ \ \vdots  $$

$$ {\bf z}_L = f_{\theta^{(L)}} ({\bf z}_{L-1}) $$


Note that ${\bf z}_l \in \mathbb{R}^{10 \times D}$, for $l=0, 1,2, \cdots, L$,

$$ 
{\bf z}_l = \left[ 
  \begin{matrix} 
  && \textcolor{red}{\bf z}_l^0 &&\\
  && \textcolor{green}{\bf z}_l^1 &&\\
  &&  \vdots &&\\
  && {\bf z}_{l}^{9} &&
  \end{matrix}
\right] 
$$

where 
${\bf z}_l^i$ is a $1 \times D$ vector.

Notice how the transformer transforms the input ${\bf z}_0$ to the output ${\bf z}_L$:

$$ 
\left[ 
  \begin{matrix} 
  && \textcolor{red}{\bf z}_0^0 && \\
  && \textcolor{green}{\bf z}_0^1 &&\\
  &&  \vdots &&\\
  && {\bf z}_{0}^{9} &&
  \end{matrix}
\right] 
\rightarrow
\left[ 
  \begin{matrix} 
  && \textcolor{red}{\bf z}_1^0&&\\
  && \textcolor{green}{\bf z}_1^1 &&\\
  &&  \vdots &&\\
  && {\bf z}_{1}^{9} &&
  \end{matrix}
\right] 
\rightarrow
\left[ 
  \begin{matrix} 
  && \textcolor{red}{\bf z}_2^0 &&\\
  && \textcolor{green}{\bf z}_2^1 &&\\
  &&  \vdots &&\\
  && {\bf z}_{2}^{9} &&
  \end{matrix}
\right] 
\rightarrow
\cdots
\rightarrow
\left[ 
  \begin{matrix} 
  && \textcolor{red}{\bf z}_{L-1}^0 &&\\
  && \textcolor{green}{\bf z}_{L-1}^1 &&\\
  &&  \vdots &&\\
  && {\bf z}_{L-1}^{9} &&
  \end{matrix}
\right] 
\rightarrow
\left[ 
  \begin{matrix} 
  && \textcolor{red}{\bf z}_L^0 &&\\
  && \textcolor{green}{\bf z}_L^1 &&\\
  &&  \vdots &&\\
  && {\bf z}_{L}^{9} &&
  \end{matrix}
\right] 
$$

The input ${\bf z}_0$ and the output ${\bf z}_L$ have the same dimension, 
and ${\bf z}_0^i$ in ${\bf z}_0$ eventually becomes ${\bf z}_L^i$ in ${\bf z}_L$


### Classification Tasks

Recall that $${\bf x}_{class}$$ is the top row of ${\bf z}_0$ denoted by ${\bf z}_0^0$. 
The patch embedding in ${\bf z}_L$ corresponding to 
$${\bf x}_{class}$$ is its top row ${\bf z}_L^0$.

In Eq (4) of [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929), ${\bf z}_L^0$ is used for classification tasks. 
Alternatively, "mean-pooling" of the patch embeddings 
$$\frac{1}{10}\sum_{i=0}^{9} {\bf z}_L^i$$ 
can be used.

<center width="100%"><img src="/assets/img/vit/classification_on_cls.png" width="200px"></center>


In [Phill Wang's implementation](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py),

```python
  # x == z_0
  x = self.transformer(x) # (batch_size, 10, D)
  # x == z_L
  x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

  return self.mlp_head(x)
```