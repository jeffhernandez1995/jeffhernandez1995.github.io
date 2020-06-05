---
title: "Design of experiments (DoE) in machine learning research"
author: Jefferson Hernández
categories: design_of_experiments, machine_learning, english
date:   2020-05-04 18:09:00
use_math: true
---

In this post, we will study the applicability of design of experiments (DoE) in machine learning (ML) experiments, to do so we will use a machine learning paper as a case study. All the code necessary to reproduce this blog can be found on [here](https://github.com/jeffhernandez1995/fastweights).

## Motivation
Machine learning is a kind of a special science field, while sometimes unnecessary mathematical and theoretical and many times rightfully so, the majority of machine learning progress is driven by experiments, either by proposing new architectures, new optimizers, or new paradigms of training. As a mainly experimental field it is of extreme importance to carry out empirically rigorous experiments to support the claims that cannot be proven mathematically and to make sure that results cannot be explained by alternative hypotheses other than our own.

Recently, there have been concerns about the reproducibility or applicability of a significant proportion of published research. Some of these include the work of Henderson et. al. [\[7\]](#henderson2018deep) that concludes that recent advancements in deep reinforcement learning might not be significant due to problems in experimental design and evaluation. Likewise the work of Musgrave et. al. [\[8\]](#musgrave2020metric) which has cast doubt at advancements in metric learning caused by several flaws in experiment design, unfair comparisons, and testing set leakage. These are not isolated events as it is easy to find other examples in other sub-areas of ML [\[9](#lucic2018gans), [10\]](#melis2017state).

These problems are not unique to ML research, in fact fields such as psychology where lack of experimental rigor that have been going for years have eroded trust in the entire field [\[11\]](#open2015estimating). In ML, these trends are recent and have been identified by some scholars (shamefully taken from this excellent paper [\[6\]](#lipton2018troubling) from professor Zachary Lipton), these are:

1. Failure to distinguish between explanation and speculation.
2. Failure to identify the sources of empirical gains, e.g. emphasizing unnecessary modifications to neural architectures when gains actually stem from hyper-parameter tuning.
3. Mathiness: the use of mathematics that obfuscates or impresses rather than clarifies, e.g. by confusing technical and non-technical concepts.
4. Misuse of language, e.g. by choosing terms of art with colloquial connotations or by overloading established technical terms.

Int this post we only focus on trends 1 and 2.

## Disclaimer
The fact that these problems have been identified by people from within the community is itself a great step. Likewise many researchers have already started suggesting solution to these problems by proposing workflows for lab collaboration and reproducibility [\[12\]](#chen2019open), by proposing widespread use of statical tools like ablation studies  [\[6\]](#lipton2018troubling), significance testing and other statistical methods. In some sense, this blog post can be seen as a lightweight version of the work of Boquet et. al. [\[13\]](#boquet2019decovac) (if you have the time please read this paper), where they propose the use of design of experiments (DoE) which can help address the failure to identify the factors of empirical variation and failure to distinguish between explanation and speculation highlighted by [\[6\]](#lipton2018troubling). Unlike them, I only choose one paper [\[1\]](#ba2016using) to provide a concrete example, while I understand that the results of presented in this blog can be seen as an attack on the authors and I do stand by the claims made here, it is not my intention to in any way attack the authors of said paper. Finally, I try to end on the positive note, namely that this work has inspired more research [\[2](#zhang2017learning), [3](#schlag2018learning), [4\]](#le2020self) tipping the balance of knowledge in science in a positive way.

## Case Study (Using Fast Weights to Attend to the Recent Past [\[1\]](#ba2016using))

### What are _fast weights_?
Fast Weights are extend standard vanilla recurrent neural network architecture with an associative memory. In the context of this paper, the authors identify two types of memory in traditional recurrent neural networks, hidden activity vectors $h_t$, that are updated every time-step, and serve as short-term memory and slow weights (traditional weights matrices) that are updated at the end of a batch and that have more memory capacity. The authors motivate a third type of memory called fast weights  that has much higher storage capacity than the neural activities but much faster dynamics than the standard slow weights [\[1\]](#ba2016using). (We note as the author did that these concepts were developed much early in [\[14\]](#hinton1987using) and [\[15\]](#schmidhuber1992learning))

The author also give biological motivations for the concept of fast weights, namely that human do not store exact patterns of neural activity as memory, instead memory retrieval involves reconstructing neural patterns through a set of associative weights which can map to many other memories as well. 
## _References_

<a name="ba2016using"></a> [\[1\]](#ba2016using) Ba, J., Hinton, G. E., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016). **Using fast weights to attend to the recent past**. In Advances in Neural Information Processing Systems (pp. 4331-4339).

<a name="zhang2017learning"></a> [\[2\]](#zhang2017learning) Zhang, W., & Zhou, B. (2017). **Learning to update auto-associative memory in recurrent neural networks for improving sequence memorization**. arXiv preprint arXiv:1709.06493.

<a name="schlag2018learning"></a> [\[3\]](#schlag2018learning) Schlag, I., & Schmidhuber, J. (2018). **Learning to reason with third order tensor products**. In Advances in neural information processing systems (pp. 9981-9993).

<a name="le2020self"></a> [\[4\]](#le2020self) Le, H., Tran, T., & Venkatesh, S. (2020). **Self-Attentive Associative Memory**. arXiv preprint arXiv:2002.03519.

<a name="orhan2017note"></a> [\[5\]](#orhan2017note) Orhan, E. (2017). **A note on fast weights: do they really work as advertised?**. [url](https://severelytheoretical.wordpress.com/2017/10/14/a-note-on-fast-weights-do-they-really-work-as-advertised/).

<a name="lipton2018troubling"></a> [\[6\]](#lipton2018troubling) Lipton, Z. C., & Steinhardt, J. (2018). **Troubling trends in machine learning scholarship**. arXiv preprint arXiv:1807.03341.

<a name="henderson2018deep"></a> [\[7\]](#henderson2018deep) Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018, April). **Deep reinforcement learning that matters**. In Thirty-Second AAAI Conference on Artificial Intelligence.

<a name="musgrave2020metric"></a> [\[8\]](#musgrave2020metric) Musgrave, K., Belongie, S., & Lim, S. N. (2020). **A metric learning reality check**. arXiv preprint arXiv:2003.08505.

<a name="lucic2018gans"></a> [\[9\]](#lucic2018gans) Lucic, M., Kurach, K., Michalski, M., Gelly, S., & Bousquet, O. (2018). **Are gans created equal? a large-scale study**. In Advances in neural information processing systems (pp. 700-709).

<a name="melis2017state"></a> [\[10\]](#melis2017state)  Melis, G., Dyer, C., & Blunsom, P. (2017). **On the state of the art of evaluation in neural language models**. arXiv preprint arXiv:1707.05589.

<a name="open2015estimating"></a> [\[11\]](#open2015estimating) Open S1ience Collaboration. (2015). **Estimating the reproducibility of psychological science**. Science, 349(6251), aac4716.

<a name="chen2019open"></a> [\[12\]](#chen2019open) Chen, X., Dallmeier-Tiessen, S., Dasler, R., Feger, S., Fokianos, P., Gonzalez, J. B., ... & Rodriguez, D. R. (2019). **Open is not enough**. Nature Physics, 15(2), 113-119.

<a name="boquet2019decovac"></a> [\[13\]](#boquet2019decovac) Boquet, T., Delisle, L., Kochetkov, D., Schucher, N., Atighehchian, P., Oreshkin, B., & Cornebise, J. (2019). **DECoVaC: Design of Experiments with Controlled Variability Components**. arXiv preprint arXiv:1909.09859.

<a name="hinton1987using"></a> [\[14\]](#hinton1987using) Hinton, G. E., & Plaut, D. C. (1987, July). Using fast weights to deblur old memories. In Proceedings of the ninth annual conference of the Cognitive Science Society (pp. 177-186).

<a name="schmidhuber1992learning"></a> [\[15\]](#schmidhuber1992learning) Schmidhuber, J. (1992). Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1), 131-139.

{% include disqus.html %}