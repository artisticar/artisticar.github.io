---
title: PAC in a nutshell
date: 2020-02-11 22:53:23
tags:
- Machine Learning
- Math
thumbnail: /gallery/kq.jpg
---



# **Probably Approximately Correct**

Notations:

$H$ : Class of hypothesis

$D$ : Distribution

$f$ : Labeling function of $D$ 

$S$ : Sample

$m$ : Sample size

$L$ : Loss function

---

**Empirical Risk Minimization(ERM)**:

The principle behind ERM is to find a predictor $h$ that minimizes empirical the risk $L_S(h)$

<br>

When the **realizability** assumption holds, the following conditions are met

- $\exists h_* \in H$ such that $L_{D,f}(h_*) = 0$
- This implies that there exists a hypothesis that correctly predicts all the labels according to the distribution $D$

*Corollary*: $P(L_S(h*) = 0) = 1$, for any $S$ sampled from $D$

Under this assumption, the predictor $h$ returned by ERM will always have an empirical risk $L_S(h)$ = 0.

<br>

Assuming the sample $S$ is sampled from the distribution $D$ following the i.i.d assumption.

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

<br>

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

<br>

---

<br>

In cases where $H$ is finite, we can establish a lowerbound on $m$ given the condifence parameter $\delta$ and accuracy parameter $\epsilon$. 

If we aim to have at most $\delta $ odds of obtaining a misleading hypothesis — **i.e.** at least $1 - \delta $ confidence of getting a hypothesis that's approximately correct($L_{(D,f)}(h) < \epsilon$), we have

- $1 - \vert H\vert$ $e^{-\epsilon m}$ $>= 1 - \delta$ , which simplifies to
- $m \geq \frac{log({\vert H\vert}/{\delta})}{\epsilon}$

We define $m_H (\epsilon, delta)$ as **sample complexity**

- $m_H (\epsilon, \delta) \leq \frac{log({\vert H\vert}/{\delta})}{\epsilon}$ 

This represents the minimum number of examples required for any ERM learned from an i.i.d sample to be:

- **Probably**(with confidence $> 1 - \delta$) 
- **Approximately** ($L_{D,f}(h) < \epsilon$) 
- **Correct**

This is the origin of the term **PAC**.

<br>

---

<br>

Suppose the realizability assumption holds for $H, D, f$, and $m >= m_H(\epsilon, \delta)$ i.i.d. examples are sampled from $D$.

Then, $\forall$ $\epsilon$ and $\delta$ , if a learning algorithm(**e.g.** ERM) return a hypothesis $h$ such that with confidence $1 - \delta$, its $L_{D,f}(h) < \epsilon$, we say that the hypothesis class $H$ is PAC learnable. 

*Corollary*: Every finite class hypothesis is PAC learnable with sample complexity

- $m_H (\epsilon, \delta) <= \frac{log({\vert H\vert}/{\delta})}{\epsilon}$ 

It's important to note that this doesn't imply infinite hypothesis classes are unlearnable.  Some infinite classes, such as axis-aligned rectangles and concentric circles(often cited in learning theory exercises), are examples of PAC-learnable classes. 

---

## **Agnostic PAC learnable**

You know how some people claim they can determine if a watermelon is tasty based on its weight and diameter? While it's highly unlikely to find two identical watermelons, theoretically, you could have watermelons with the same weight and diameter that taste entirely different.

I guess this is not a perfect example because of the subjectivity of 'tastiness'', but let's assume that there's a universal standard for it. In that case, if we know everything about the watermelon, everything about every single piece of physical matter it is composed of, then I guess there exist a hypothesis that can generate perfect prediction anytime. But let's be realistic; this is virtually impossible because the features we have are usually **not deterministic**.

In such scenarios, no single hypothesis can provide a definitive correct answer. The best we can hope for is a hypothesis that gives a **probability distribution** based on those features.

For example, given that the watermelon weigh 12 kilos and have a diameter of 30 centimeters, most accurate prediction might be that there's a 70% chance it's tasty and a 30% chance it's not. This isn't just a random guess; it's based on observed data and what we call the **Bayes Optimal Predictor**.

It's clear that the realizability assumption doesn't hold here, because

- $min_{h\in H} L_D(h)$ > 0

We also want to introduce new definitions for empirical error and true error:

**New Empirical Error:**

- $L_S(h) = \frac{\vert \{i \vert h(x_i) \neq y_i)\}\vert}{m}$
- Compared to the original definition of empirical error $L_S(h) = \frac{\vert \{i \vert h(x_i) \neq h_*(i)\}\vert}{m}$ , we see that the $h_*(i)$ is replaced by $y_i$. This is because there's no longer an $h_*(i)$ that can make perfect predictions. 

**New True Error:**

- $L_D(h) = P_{(x,y)\sim D} [h(x) \neq y]$
- Notice that the label $y$ is also drawn from the distribution, making it probablistic. 

<br>

In this broader context of prediction tasks, we introduce the concept of **Agnostic PAC Learnability**:

The definition mirrors that of standard PAC learnability but adds an extra term  $min_{h\in H}L_D(h)$ into the inequality of PAC.

- $P(L_{D,f}(h) -min_{h\in H} L_D(h) \leq \epsilon)$ $\geq 1-\delta$
- The only addition is the $min_{h\in H}L_D(h)$ term
- Our objective now shifts to getting as close as possible to the accuracy of the Bayes Optimal Predictor, rather than aiming for an unrealistic 0% error rate.

<br>

So, the next time someone who claims to be a watermelon expert picks out a dud for you at Trader Joe's, it might not be because they're fibbing. They could very well be making the most informed choice possible but just got unlucky this time around. 

<br>

---

<br>

### **Steps to add LaTeX support to your Jekyll blog:**

Add the following code in the file **_includes/head.html**

```html
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 
```

Reference $\vert$ https://stackoverflow.com/questions/26275645/how-to-support-latex-in-github-pages

<br>

**Side notes**

First time writing in Latex,  so I figured I'd put together this blog to get the hang of it.

---

*Shai Shalev-Shwartz and Shai Ben-David. 2014.* *Understanding Machine Learning: From Theory to Algorithms*. *Cambridge University Press, USA.*

