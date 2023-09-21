---
title: PAC in a nutshell
date: 2020-02-11 22:53:23
tags:
- Machine Learning
- Math
thumbnail: /gallery/kq.jpg
---



# **Probably Approximately Correct**

---

Notations:

$H$ : Class of hypothesis

$D$ : Distribution

$f$ : Labeling function of $D$ 

$S$ : Sample

$m$ : Sample size

$L$ : Loss function

<br>

---

**Empirical Risk Minimization(ERM)**:

The principle behind ERM is to find a predictor $h$ that minimizes empirical the risk $L_S(h)$. 

<br>

When the **realizability** assumption holds, the following conditions are met

- $\exists h_* \in H$ such that $L_{D,f}(h_*) = 0$
- This implies that there exists a hypothesis that correctly predicts all the labels according to the distribution $D$

*Corollary*: $P(L_S(h*) = 0) = 1$, for any $S$ sampled from $D$

Under this assumption, the predictor $h$ returned by ERM will always have an empirical risk $L_S(h)$ = 0.

<br>

The examples in the samples are sampled following the i.i.d assumption.

**i.i.d. assumption**:

- i.i.d. stands for "independently and identically distributed"
- **i.e.** all the examples in the sample $S$ are i.i.d. with respect to the distribution $D$
- we denote this as $S$ ~ $D^m$

<br>

Even if the i.i.d. assumption holds, It's still possible to get non-representative samples from the distribution $D$. In such cases,  ERM may achieve zero error on the training set $S$but perform poorly on the actual distribution $D$. To account for this, we introduce a ***confidence parameter*** of tolerance: 

- $\delta$ , which denotes the probability of getting a bad sample. 

Similarly, even with a representative sample $S$, the sample may not capture the entire distribution $D$. For this, we introduce an ***accuracy parameter*** of tolerance:

- $\epsilon$ , which denotes the error rate of the hypothesis in the true distribution $D$.

<br>

---

We aim to upper-bound the probability of selecting a **misleading hypothesis**. A misleading hypothesis achieves zero error on the training set $S$ but has an error rate $>$ $\epsilon$ on the distribution $D$.

### **Proof**

We define $H_B$ as the set of hypothesis that has an error $>$ $\epsilon$ on the distribution $D$.

- $H_B$ = {$h \in H : L_{(D,f)}(h) > \epsilon$}

Therefore, the probability of getting a misleading hypothesis from the sample $S$ is 

- $D^m$($\cup_{h\in H_B}${$S\vert_x: L_S(h) = 0$})

Apply the union bound to this expression, we we find that the above probability is upperbounded by 

- $\sum_{h\in H_B}$ $D^m$({$S\vert_x: L_S(h) = 0$})

The probability of obtaining a single bad hypothesis $h$ is:

- $(1 - L_{(D,f)}(h))^m$ $\leq (1 - \epsilon)^m$ $<= e^{-\epsilon m}$
- Here, $1 - L_{(D,f)}(h)$ is the probability of get an individual misleading example for $h$

Substituting this bound into the sum gives:

- $\vert H_B\vert$ $e^{-\epsilon m}$
- Since $\vert H_B\vert$ (the number of bad hypotheses) is not known, we upper-bound it by the total number of hypotheses, $\vert H\vert$.

Putting all these observations together, we find that the upper-bound is:

- $D^m$($\cup_{h\in H}${$S\vert_x: L_{(D,f)}(h) > \epsilon,$ $L_S(h) = 0$}) $\leq \vert H\vert$ $e^{-\epsilon m}$

---

In the case that $H$ is finite, we can apply a lowerbound on $m$ given the condifence parameter $\delta$ and accuracy parameter $\epsilon$. 

If we want to have at most $\delta $ odds on getting a misleading hypothesis

**i.e.** at least $1 - \delta $ confidence of getting a hypothesis that's approximately correct($L_{(D,f)}(h) < \epsilon$)

- $1 - \vert H\vert$ $e^{-\epsilon m}$ $>= 1 - \delta$ , which gives the result below
- $m \geq \frac{log({\vert H\vert}/{\delta})}{\epsilon}$

We define $m_H (\epsilon, delta)$ as **sample complexity**

- $m_H (\epsilon, \delta) \leq \frac{log({\vert H\vert}/{\delta})}{\epsilon}$ 

which denotes the minimum number of examples for any ERM learned from an i.i.d sample to be

- **probably**(with confidence $> 1 - \delta$) **approximately** ($L_{D,f}(h) < \epsilon$) **correct**.

This is where **PAC** comes from.

---

Assume that the realizability assumption holds for $H, D, f$

and $m >= m_H(\epsilon, \delta)$ i.i.d. examples are sampled from $D$.

Then if given $\forall$ $\epsilon$ and $\delta$ , a learning algorithm(**e.g.** ERM) return a hypothesis $h$ such that with confidence $1 - \delta$, its $L_{D,f}(h) < \epsilon$, we say that it is PAC learnable. 

*Corollary*: Every finite class hypothesis is PAC learnable with sample complexity

- $m_H (\epsilon, \delta) <= \frac{log({\vert H\vert}/{\delta})}{\epsilon}$ 

This doesn't mean that infinite classes aren't learnable. Some, such as axis-aligned rectangles and concentric circles in the exercises of the book are examples of the learnable ones. 

---

## **Agnostic PAC learnable**

Some people claim that they know how to see if a watermelon is tasty or not really well. Let's say they evaluate, or, make the prediction based on the following two features - weight and diameter. Though in reality it's impossible to have two identical watermelons, it's theoretically possible that some watermelons have the same numerical value of the two features, yet turn out that one is tasty while the other is not. Think about it in real life. Two watermelons can look really similar but taste very differently. 

I guess this is not a perfect example because tastiness is subjective, but let's assume that there's a universal standard of measuring tastiness. In that case, if we know everything about the watermelon, everything about every single piece of physical matter it is composed of, then I guess there exist a hypothesis that can generate perfect prediction anytime. However, this is obviously impossible, and the features given to us are usually **not deterministic**.

Then there's no hypothesis that can give a correct definite answer. All the hypotheses can do is to give a **distribution** given the input features. 

For example, given that the watermelon weigh 12 kilos and have a diameter of 30 centimeters, the best possible prediction is to say that it has a 70% chance of being tasty and 30% chance that it tastes bad. This predictor is based on the labels on the distribution, and we can't do better than that. We call this predictor **The Bayes Optimal Predictor**.

We can clearly see that the realizability assumption doesn't hold here, because

- $min_{h\in H} L_D(h)$ > 0

We also want to define a new empirical error and a new true error:

**New Empirical Error:**

- $L_S(h) = \frac{\vert \{i \vert h(x_i) \neq y_i)\}\vert}{m}$
- Comparing to the original definition of empirical error $L_S(h) = \frac{\vert \{i \vert h(x_i) \neq h_*(i)\}\vert}{m}$ , we see that the $h_*(i)$ is replaced by $y_i$ because there's no $h_*(i)$ that predicts everything right anymore. 

**New True Error:**

- $L_D(h) = P_{(x,y)\sim D} [h(x) \neq y]$
- Notice that $y$ , which is the label, is also drawn from the distribution, which is probablistic. 

<br>

In this more general case of prediction task, we want to define **Agnostic PAC Learnability**:

The definition is exactly the same as PAC learnability, except that we introduce an extra term  $min_{h\in H}L_D(h)$ into the inequality of PAC.

- $P(L_{D,f}(h) -min_{h\in H} L_D(h) \leq \epsilon)$ $\geq 1-\delta$
- The only difference is the $min_{h\in H}L_D(h)$ term
- Our goal now is to get close to the accuracy of the Bayes Optimal Predictor but not 0 error rate which is unrealistic

<br>

So now you know, when the people who claim to be expert in watermelon got you bad watermelons from the grocery store, it may not be because of they're lying. In fact they might be making the optimal decision but just happened to be unlucky. 

---

### **Steps to add LaTeX support to your Jekyll blog:**

Add the following code in the file **_includes/head.html**

> <script type="text/x-mathjax-config">
>     MathJax.Hub.Config({
>       tex2jax: {
>         skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
>         inlineMath: [['$','$']]
>       }
>     });
>   </script>
>   <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 

Reference | https://stackoverflow.com/questions/26275645/how-to-support-latex-in-github-pages

<br>

---

*Shai Shalev-Shwartz and Shai Ben-David. 2014.* *Understanding Machine Learning: From Theory to Algorithms*. *Cambridge University Press, USA.*

