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

A principle that is to find a predictor $h$ that minimizes empirical risk $L_S(h)$. 

<br>

When **realizability** assumption holds, that is

- $\exists h_* \in H$ such that $L_{D,f}(h_*) = 0$
- This means that it's possible to have a hypothesis that correctly predicts all the labels in the distribution $D$

*Corollary*: $P(L_S(h*) = 0) = 1$, with any $S$ sampled from $D$

the predictor $h$ returned by the ERM always has an empirical risk $L_S(h)$ = 0.

<br>

And the examples in the sample are sampled following

**i.i.d. assumption**:

- i.i.d. stands for "independently and identically distributed"
- **i.e.** all the examples in the sample follow this assumption with respect to the distribution $D$
- we denote it as $S$ ~ $D^m$

<br>

As the i.i.d. assumption holds, It's always possible to get non-representative samples from the distribution. In this case though the ERM achieves zero error on the training set, it doesn't do a good job predicting over the true distribution. Therefore we would give a ***confidence parameter*** of tolerance: 

- $\delta$ , which denotes the possibility of getting a bad sample. 

However, even when we get a pretty representative sample, since it's not the whole of the distribution, it's likely that it doesn't capture all of the detail of the distribution. Again we would give an ***accuracy parameter*** of tolerance:

- $\epsilon$ , which denotes the error rate of the hypothesis in the true distribution.

<br>

---

We want to upperbound the probability of getting a **misleading hypothesis**. These hypotheses achieve 0 error on the training set but error > $\epsilon$ on the distribution $D$.

### **Proof**

We define $H_B$ as the set of hypothesis that has an error > $\epsilon$ on the distribution.

- $H_B$ = {$h \in H : L_{(D,f)}(h) > \epsilon$}

Therefore, the probability of getting a misleading hypothesis is 

- $D^m$($\cup_{h\in H_B}${$S|_x: L_S(h) = 0$})

Apply union bound to it, we get that the probability above is upperbounded by 

- $\sum_{h\in H_B}$ $D^m$({$S|_x: L_S(h) = 0$})

The error of getting an individual bad hypothesis is

- $(1 - L_{(D,f)}(h))^m$ $\leq (1 - \epsilon)^m$ $<= e^{-\epsilon m}$
- $1 - L_{(D,f)}(h)$ is the probability of get an individual misleading example

Hence, we substitute the bound we got in the inequality above to the sum to obtain

- $|H_B|$ $e^{-\epsilon m}$
- $|H_B|$ is the number of bad hypothesis which isn't known, we again upperbound it by the total number of hypothesis - $|H|$.

Putting all the things together, we have an upperbound that is

- $D^m$($\cup_{h\in H}${$S|_x: L_{(D,f)}(h) > \epsilon,$ $L_S(h) = 0$}) $\leq |H|$ $e^{-\epsilon m}$

---

In the case that $H$ is finite, we can apply a lowerbound on $m$ given the condifence parameter $\delta$ and accuracy parameter $\epsilon$. 

If we want to have at most $\delta $ odds on getting a misleading hypothesis

**i.e.** at least $1 - \delta $ confidence of getting a hypothesis that's approximately correct($L_{(D,f)}(h) < \epsilon$)

- $1 - |H|$ $e^{-\epsilon m}$ $>= 1 - \delta$ , which gives the result below
- $m \geq \frac{log({|H|}/{\delta})}{\epsilon}$

We define $m_H (\epsilon, delta)$ as **sample complexity**

- $m_H (\epsilon, \delta) \leq \frac{log({|H|}/{\delta})}{\epsilon}$ 

which denotes the minimum number of examples for any ERM learned from an i.i.d sample to be

- **probably**(with confidence $> 1 - \delta$) **approximately** ($L_{D,f}(h) < \epsilon$) **correct**.

This is where **PAC** comes from.

---

Assume that the realizability assumption holds for $H, D, f$

and $m >= m_H(\epsilon, \delta)$ i.i.d. examples are sampled from $D$.

Then if given $\forall$ $\epsilon$ and $\delta$ , a learning algorithm(**e.g.** ERM) return a hypothesis $h$ such that with confidence $1 - \delta$, its $L_{D,f}(h) < \epsilon$, we say that it is PAC learnable. 

*Corollary*: Every finite class hypothesis is PAC learnable with sample complexity

- $m_H (\epsilon, \delta) <= \frac{log({|H|}/{\delta})}{\epsilon}$ 

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

- $L_S(h) = \frac{| \{i | h(x_i) \neq y_i)\}|}{m}$
- Comparing to the original definition of empirical error $L_S(h) = \frac{| \{i | h(x_i) \neq h_*(i)\}|}{m}$ , we see that the $h_*(i)$ is replaced by $y_i$ because there's no $h_*(i)$ that predicts everything right anymore. 

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

First download Pandoc,

> brew install Pandoc

Then install hexo-renderer-pandoc,

> npm uninstall hexo-renderer-marked --save
> npm install hexo-renderer-pandoc --save

<br>

---



---

*Shai Shalev-Shwartz and Shai Ben-David. 2014.* *Understanding Machine Learning: From Theory to Algorithms*. *Cambridge University Press, USA.*

