---
title: "Why Patch Features Go Global: A First-Order View of Feature Degradation and Why Gram Anchoring Helps"
author: Jefferson Hernandez
categories: self-supervised_learning, computer_vision, machine_learning, english
date:   2025-10-10 18:09:00
use_math: true
---

Self-supervised learning has been a game-changer for training large-scale Vision Transformers (ViTs), allowing models to learn powerful visual representations without human-provided labels. However, popular methods like DINO and iBOT suffer from a curious problem: as they train longer, their ability to handle dense, local tasks degrades, even as their global performance improves. This phenomenon, known as **feature degradation**, is characterized by noisy patch similarity maps and an over-alignment of patch features with the global CLS token.

| ![Feature degradation]({{ '/pictures/feature_degradation.png' | relative_url }}) | 
|:--:| 
| **Figure 1**: Evolution of the cosine similarity between the patch noted in red and all other patches. As training progresses, the features produced by the model become less localized and the similarity maps become noisier.. Extracted from [\[1\]](#simeoni2025dinov3).|

This post dives into the mechanics of this degradation, drawing from a simple, first-order analysis. We'll explore why this happens and then discuss a solution called **Gram anchoring**, which counteracts the issue. This analysis is based on the insights from the DINOv3 paper [\[1\]](#simeoni2025dinov3). In order to understand this post, a basic familiarity with Vision Transformers, self-supervised learning, and matrix calculus is helpful.


## Setup and Notation

Let's start by defining our terms. An image $$x$$ is fed into a student backbone model $$\phi_\theta$$, which produces a set of patch features $$H\in\mathbb{R}^{P\times d}$$ (where each row $$h_i^\top$$ is a feature for a patch) and a special CLS token $$h_{\mathrm{cls}}\in\mathbb{R}^d$$.

A linear head $$W\in\mathbb{R}^{K\times d}$$ then maps any given feature $$h$$ to logits $$z=Wh$$, which are converted to probabilities $$p=\mathrm{softmax}(z)$$. Our training target $$t$$ comes from a momentum teacher network $$\bar\phi$$.

We are particularly interested in the *within-image* geometry of the patches, which we capture using the cosine Gram matrix. First, we row-normalize the patch features $$H$$ to get $$X$$, and then compute the Gram matrix $$G$$:

$$X := \text{row-normalize}(H) \in \mathbb{R}^{P\times d}, \qquad G := XX^\top \in \mathbb{R}^{P\times P}. \tag{1}\label{eq:gram}$$

The entry $$G_{ij}$$ represents the cosine similarity between the features of patch $$i$$ and patch $$j$$. This matrix is crucial because it encodes the fine spatial relationships within the image, like boundaries and distinct regions.

### Losses
We'll consider two standard loss components:

* **Global CE (DINO-style):** The cross-entropy loss is applied to a global token (like the CLS token). We'll denote the student and teacher probability distributions as $$p^{(\mathrm{g})}$$ and $$t^{(\mathrm{g})}$$.
* **iBOT masked-patch CE:** For a set of masked patches $$M$$, the student's patch distribution $$p_i$$ is trained to match the teacher's target distribution $$t_i$$.

The total loss is a simple weighted sum of these global and patch-level terms.

## First-Order Gradient Dynamics

To understand how features change during training, we'll look at a single gradient step in a linearized fashion.

### Patch CE (iBOT)
Let's define the residuals (the difference between prediction and target) as $$r_i := p_i - t_i \in \mathbb{R}^K$$. We can stack these into a matrix $$R\in\mathbb{R}^{P\times K}$$. For a small learning rate $$\eta > 0$$, the change in the patch features $$H$$ is given by:

$$\Delta H_{\mathrm{patch}} = -\eta\, R\, W. \tag{2}\label{eq:deltaH-patch}$$

### Global CE (DINO)
For the global loss, let the residual be $$r_{\mathrm{g}} := p^{(\mathrm{g})} - t^{(\mathrm{g})}\in\mathbb{R}^K$$. The update for a single patch feature $$h_i$$ depends on the Jacobian of the CLS token with respect to that patch, $$J_i := \partial h_{\mathrm{cls}}/\partial h_i$$. The update is:

$$\Delta h_i = -\eta\, J_i^\top W^\top r_{\mathrm{g}}. \tag{3}\label{eq:deltaH-global-exact}$$

Late in training, the self-attention mechanism tends to mix information broadly, causing the Jacobians to be very similar across patches ($$J_i\approx J$$). This allows us to approximate the global update for all patch features as a rank-1 matrix:

$$\Delta H_{\mathrm{global}} \approx -\eta\, \mathbf{1}\, a^\top, \tag{4}\label{eq:deltaH-global}$$

where $$a := J^\top W^\top r_{\mathrm{g}}\in\mathbb{R}^d$$ is a shared direction vector. This means every patch feature is pushed in nearly the same direction.

### Effect on the Gram Matrix
The change in the unnormalized Gram matrix $$G_H := HH^\top$$ is:

$$\Delta G_H = (\Delta H)H^\top + H(\Delta H)^\top. \tag{5}\label{eq:deltaG-raw}$$

While we technically analyze the normalized Gram matrix $$G=XX^\top$$, the core rank-based arguments that follow hold for both.

## Low-Rank Drift Under Correlated Targets

The core insight is that late in training, the **residuals become highly correlated across patches**. The global loss naturally induces this, and the sharpened teacher targets in iBOT also push different patches toward similar distributions. This correlation leads to low-rank updates to the feature matrix $$H$$.

**Lemma 1:** (Rank bound for Gram updates)
*If $$\Delta H$$ has rank $$r$$, then $$\Delta G_H$$ in equation $$\eqref{eq:deltaG-raw}$$ has rank at most $$2r$$.*

<details>
  <summary><strong>Click to see proof</strong></summary>

**Proof.** Write \(A := \Delta H\) and \(B := H\).
Then \(\Delta G_H = AB^\top + BA^\top\). Since \(\operatorname{rank}(AB^\top) \le \min\{\operatorname{rank}(A),\operatorname{rank}(B)\} \le r\) and likewise \(\operatorname{rank}(BA^\top)\le r\), we have
\[
\operatorname{rank}(\Delta G_H) \le \operatorname{rank}(AB^\top)+\operatorname{rank}(BA^\top)\le 2r.
\]
\(\square\)

</details>

**Assumption 1:** (Correlated residuals)
*The patch residual matrix $$R$$ has a low rank $$r\ll P$$ (often $$r\approx 1$$), and the Jacobians are shared ($$J_i\approx J$$).*

**Proposition 1:** (Low-rank drift & spatial homogenization)
*Under Assumption 1, the updates from both iBOT and global CE produce a low-rank $$\Delta G_H$$. Consequently, patch features are moved in parallel along a few common directions, which reduces local contrast in the Gram matrix $$G$$ and increases the alignment between the CLS token and patch features.*

In essence, the model starts to see all patches within an image as being more similar to each other than they actually are, effectively blurring the fine-grained geometric information we care about for dense tasks.

## A Simple Spectral Drift Lemma

Let's look at this from a spectral perspective. Consider a rank-1 global drift $$\Delta H=\mathbf{1}a^\top$$.

**Lemma 2:** (Energy transfer to low-frequency modes)
*For a drift $$\Delta H=\mathbf{1}a^\top$$, the Gram update is $$\Delta G_H = \mathbf{1} b^\top + b \mathbf{1}^\top$$, where $$b:=Ha$$. This update increases the Rayleigh quotient along the constant vector $$\mathbf{1}$$, meaning energy is transferred to the lowest-frequency component of the Gram matrix. This inflates the average pairwise similarity between patches.*

This confirms our previous intuition: the training dynamics concentrate energy in a few "smooth" spatial modes, wiping out the high-frequency components that encode boundaries and local details.

## Why Gram Anchoring Helps

So, how do we fix this? The proposed solution is **Gram anchoring**, a loss term that encourages the student's Gram matrix to match that of a teacher with better locality (e.g., a model from an earlier training stage).

The Gram anchoring loss is defined as:

$$\mathcal{L}_{\mathrm{Gram}}(X) := \bigl\| XX^\top - X_G X_G^\top \bigr\|_F^2. \tag{6}\label{eq:gram-loss}$$

where $$X_G$$ are the row-normalized patch features from the teacher.

This loss has two powerful properties:

1.  **Orthogonal Invariance:** The term $$XX^\top$$ is invariant to any rotation of the feature space ($$X\mapsto XR$$ for $$R\in \mathrm{O}(d)$$). This is critical because it means the Gram loss *only* constrains the pairwise geometry of the patches, leaving the model free to learn global semantics without interference.
2.  **Exact Gradient:** The gradient is straightforward to compute:

$$ \frac{\partial \mathcal{L}_{\mathrm{Gram}}}{\partial X} = 4\, (XX^\top - X_G X_G^\top)\, X. \tag{7}\label{eq:gram-grad}$$

**Proposition 2:** (Complementarity with CE/iBOT)
*The low-rank drifts induced by CE/iBOT change the Gram matrix $$XX^\top$$ along a small set of smooth spatial modes. The Gram loss penalizes exactly these changes while leaving global rotations free. Therefore, Gram anchoring can repair locality (improving dense task performance) without significantly impeding the learning of global semantics.*

## What to Measure (and Why It Should Move)

This analysis suggests a few key metrics to monitor during training:

1.  **CLS–patch alignment:** The average cosine similarity between $$h_{\mathrm{cls}}$$ and the patch features $$h_i$$. This should increase as locality degrades and stabilize once Gram anchoring is applied.
2.  **Residual correlation:** For iBOT, the singular value decomposition (SVD) of the residual matrix $$R$$ should show its rank decreasing late in training. Gram anchoring should reverse this.
3.  **Spectral collapse of G:** The fraction of energy in the top few eigenvectors of the Gram matrix $$G$$ will increase during degradation. Gram anchoring should halt this.
4.  **Loss sensitivity:** With Gram anchoring, the iBOT/patch loss should decrease faster, while the global CE loss should be largely unaffected.

Sadly we don't have access to the intermediate checkpoints for the DINOv3 models to show these metrics, but if you are interested in this kind of analysis, and have enough compute to train a model from scratch, it would be interesting to see these metrics in action.

## Conclusion

The core problem is that correlated targets from global and patch-level self-supervised objectives lead to low-rank gradient updates. These updates homogenize the within-image patch geometry, destroying local detail and causing feature degradation.

**Gram anchoring** provides a targeted fix. By penalizing deviations from a "good" teacher Gram matrix, it counteracts these specific low-rank, geometry-destroying updates. Its rotational invariance ensures it doesn't interfere with the learning of high-level global semantics, making it an effective and complementary regularizer.

### Minimal recipe
Use an early-stage or high-resolution teacher to define the target Gram matrix $$X_G X_G^\top$$. Add the Frobenius norm loss $$\|XX^\top - X_GX_G^\top\|_F^2$$ for global image crops, and monitor CLS-patch alignment and the spectrum of $$G$$. If they stabilize while your dense task metrics improve, you're on the right track.

## _References_

<a name="simeoni2025dinov3"></a> [\[1\]](#simeoni2025dinov3) Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., ... & Bojanowski, P. (2025). Dinov3. arXiv preprint arXiv:2508.10104..


## Appendix: Deriving $$\partial\mathcal{L}_{\mathrm{Gram}}/\partial X$$

For those interested in the matrix calculus, here's a quick derivation.
Let $$F(X) := XX^\top - X_GX_G^\top$$. The loss is $$\mathcal{L}=\|F(X)\|_F^2 = \mathrm{tr}(F^\top F)$$. The differential is $$d\mathcal{L}=2\,\mathrm{tr}(F^\top\, dF)$$.
Since $$d(XX^\top)=dX\,X^\top + X\, dX^\top$$, we have:
$$d\mathcal{L} = 2\,\mathrm{tr}\Bigl(F^\top(dX\,X^\top + X\, dX^\top)\Bigr) = 2\,\mathrm{tr}\bigl(X^\top F^\top dX\bigr) + 2\,\mathrm{tr}\bigl(F^\top X\, dX^\top\bigr).$$
Using the cyclic property of the trace and the symmetry of $$F$$, we get:
$$d\mathcal{L} = 4\, \mathrm{tr}\bigl((FX)^\top dX\bigr) \;\Rightarrow\; \frac{\partial \mathcal{L}}{\partial X} = 4\, F(X)\,X = 4\,(XX^\top - X_GX_G^\top)X.$$
