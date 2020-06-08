---
title: "Design of experiments (DoE) in machine learning research"
author: Jefferson Hernández
categories: design_of_experiments, machine_learning, english
date:   2020-05-04 18:09:00
use_math: true
---

In this post, we will study the applicability of design of experiments (DoE) in machine learning (ML) experiments, to do so we will use a machine learning paper as a case study. I assume that the reader is familiar with RNN's. All the code necessary to reproduce these experiments can be found on [here](https://github.com/jeffhernandez1995/fastweights).

## Motivation
Machine learning (ML) is kind of a special science field, while sometimes unnecessary mathematical and theoretical and many times rightfully so, the majority of machine learning progress is driven by experiments, either by proposing new architectures, new optimizers, or new paradigms of training. As a mainly experimental field, it is of extreme importance to carry out rigorous experiments to support the claims that cannot be proven mathematically and to make sure that results cannot be explained by alternative hypotheses other than our own.

Recently, there have been concerns about the reproducibility or applicability of a significant proportion of published research in ML. Some of these include the work of Henderson et. al. [\[7\]](#henderson2018deep) that concludes that recent advancements in deep reinforcement learning might not be significant due to problems in experimental design and evaluation. Likewise the work of Musgrave et. al. [\[8\]](#musgrave2020metric) which has cast doubt at advancements in metric learning caused by several flaws in experiment design, unfair comparisons, and testing set leakage. These are not isolated events as it is easy to find other examples in other sub-areas of ML [\[9](#lucic2018gans), [10\]](#melis2017state).

These problems are not unique to ML research, in fact fields such as psychology where lack of experimental rigor that have been going for years have eroded trust in the entire field [\[11\]](#open2015estimating). In ML, the trends driving these problems are recent and have been identified by some scholars (shamefully taken from this excellent paper [\[6\]](#lipton2018troubling) from professor Zachary Lipton); these are:

1. Failure to distinguish between explanation and speculation.
2. Failure to identify the sources of empirical gains, e.g. emphasizing unnecessary modifications to neural architectures when gains actually stem from hyper-parameter tuning.
3. Mathiness: the use of mathematics that obfuscates or impresses rather than clarifies, e.g. by confusing technical and non-technical concepts.
4. Misuse of language, e.g. by choosing terms of art with colloquial connotations or by overloading established technical terms.

Int this post we only focus on trends 1 and 2.

## Disclaimer
The fact that these problems have been identified by people from within the community is itself a great step. Likewise, many researchers have already started suggesting solutions to these problems by proposing workflows for lab collaboration and reproducibility [\[12\]](#chen2019open), by proposing widespread use of statistical tools like ablation studies  [\[6\]](#lipton2018troubling), significance testing and other statistical methods. In some sense, this blog post can be seen as a lightweight version of the work of Boquet et. al. [\[13\]](#boquet2019decovac) (if you have the time please read this paper), where they propose the use of design of experiments (DoE) which can help address the failure to identify the factors of empirical variation and failure to distinguish between explanation and speculation highlighted by [\[6\]](#lipton2018troubling). Unlike them, I only choose one paper [\[1\]](#ba2016using) to provide a concrete example, while I understand that the results of presented in this blog can be seen as an attack on the authors and I do stand by the claims made here, it is not my intention to, in any way attack the authors of said paper. Finally, I try to end on the positive note, namely that this work has inspired more research [\[2](#zhang2017learning), [3](#schlag2018learning), [4\]](#le2020self) tipping the balance of knowledge in science in a positive way.

## Case Study: Using Fast Weights to Attend to the Recent Past [\[1\]](#ba2016using)

### What are _fast weights_?
Fast Weights extend standard vanilla recurrent neural network architectures with an associative memory. In the context of this paper, the authors identify two types of memory in traditional recurrent neural networks, hidden activity vectors $$\mathbf{h}_t$$, that are updated every time-step, and serve as short-term memory and slow weights (traditional weights matrices) that are updated at the end of a batch and that have more memory capacity. The authors motivate a third type of memory called fast weights  that has much higher storage capacity than the neural activities but much faster dynamics than the standard slow weights [\[1\]](#ba2016using). (We note, as the author did, that these concepts were developed much early in [\[14\]](#hinton1987using) and [\[15\]](#schmidhuber1992learning))

The author also give biological motivations for the concept of fast weights, namely that human do not store exact patterns of neural activity as memory, instead memory retrieval involves reconstructing neural patterns through a set of associative weights which can map to many other memories as well.

| ![Fast weights](https://raw.githubusercontent.com/jeffhernandez1995/jeffhernandez1995.github.io/master/notebooks/fast_weights/fast_weights_diagram.svg) | 
|:--:| 
| **Figure 1**:  The fast associative memory model. Extracted from [\[1\]](#ba2016using).|

Figure 1 shows a diagram of how fast weights affects hidden activity vector. After hidden activity $$\mathbf{h}_t$$ is computed a brief iterative settling process (of size $$S$$) is started, during this process a fast weight matrix $$\mathbf{A}$$ is updated using a form of Hebbian short-term synaptic plasticity (outer product)

$$
    \begin{aligned}
        \mathbf{A}_t = \lambda \mathbf{A}_{t-1} + \eta \mathbf{h}_t \mathbf{h}_t^\intercal,
    \end{aligned}
    \tag{1}\label{1}
$$

where $$\lambda$$ and $$\eta$$ are called decay rate and fast learning
rate respectively. $$\mathbf{A}_t$$ (assumed to be zero at the start of the sequence) maintains a dynamically changing short-term memory of the recent history of hidden activities in the network.

The next hidden activity is computed unrolling an _inner loop_ of size $$S$$ that progressively changes the hidden state (red path in Figure 1) using the input $$x_t$$ and the previous hidden vector. At each iteration of the inner loop, the fast weight matrix is exactly equivalent to attention mechanism between past hidden vectors and the current hidden vector, weighted by a decay factor [\[1\]](#ba2016using). The final equation for the model is

$$
    \begin{aligned}
        \mathbf{h}_{t+1} = f\left(\mathcal{LN}\left[\mathbf{W}_h \mathbf{h}_{t} + \mathbf{W}_x \mathbf{x}_{t} + (\eta \sum_{\tau=1}^{\tau=t-1} \lambda^{t - \tau -1} f(\mathbf{W}_h \mathbf{h}_{t} +  \mathbf{W}_x \mathbf{x}_{t})\right]\right)
    \end{aligned}
    \tag{2}\label{2}
$$

where $$\mathcal{LN}(.)$$ refers to layer normalization (LN) and the unrolling is run for $$S=1$$ steps.

### Main claim of the fast weights paper

Using four experiments the authors try to justify the advantages of fast weights over traditional recurrent architectures. These experiments are associative retrieval, MNIST classification using visual glimpses, Facial expression recognition using visual glimpses and reinforcement learning. Results of these experiments seem to suggest that the incorporated fast weight matrix is the sole responsible for the observed superior performance. However, there are two factors of variation not accounted for in the paper by Ba et. al. [\[1\]](#ba2016using). I am not the first person to identity these factors in fact researcher Emin Orhan is the first to identify these problems in [\[5\]](#orhan2017note) (His blog is great, you should definitely check it out). These factors are:
1.  As proposed in equation $$\eqref{2}$$ the model has more depth than standard recurrent architectures. In [\[5\]](#orhan2017note) Orhan noted that as proposed this architecture is not biologically plausible and that there are ways to incorporate the fast weight matrix without increasing the effective depth. This is in fact how the original fast weights were proposed in [\[14\]](#hinton1987using).
2. Layer normalization has been shown to improve the performance of vanilla recurrent networks and no classical RNN with layer normalization  or fast weight RNN without are tested in the paper, this implies that some of the improvement is due to LN.
3. Ba et. al. [\[1\]](#ba2016using) hypothesize that fast weights allows RNN's to use their recurrent units more effectively, allowing to reduce the hidden vector size without harming performance. To show this the authors compare with an LSTM, but the comparison should be carried out using standards RNN to see if the performance gains are not due to factors (1) and (2) or better initialization schemes, or the use of the optimizer.

### The role of Design of experiments (DoE)

Quoting the work of [\[13\]](#boquet2019decovac):

>We control almost completely the environment where the experiments are run and thus the data-generating process, we can define a specific **design** to reason about statistical reproducibility while comparing the results of different runs of different algorithms. 
>

For our purposes, this design is the result of having formulated three hypotheses about different factor that could explain the superior performance seen in [\[1\]](#ba2016using) rather than the fast weight matrix. We turn to Design of experiments (DoE) for a framework that will allow us to test these hypotheses. More specifically we will perform a $$2^k r$$ factorial designs with replications where $$k$$ is the number of factors and $$r$$ the number of replications. In our simple case we will assume that our observations are i.i.d, this turns the problem of estimating the effects of each factor into a linear regression model with a binary explanatory variable

$$
    \begin{aligned}
        \mathbf{y} = \mathbf{X}^\intercal \mathbf{q} + \mathbf{q_0} + \epsilon 
    \end{aligned}
    \tag{3}\label{3}
$$

where $$\mathbf{y}$$ is vector of responses, $$\mathbf{X}$$ is a binary matrix that encodes the factors and their iterations (linear, quadratic, cubic), $$\mathbf{q}$$ and $$\mathbf{q_0}$$ are called fixed effects and $$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$ is random noise. We the must perform experiments for all factor combinations (and interactions), namely $$2^k$$.

For our case study these factors are:
1. **DEPTH**: a binary variable that represent whether we use $$\eqref{2}$$ which increases the overall depth of the network or the following:
$$
    \begin{aligned}
        \mathbf{h}_{t+1} = f\left(\mathcal{LN}\left[ \left(\mathbf{W}_h  + \eta \sum_{\tau=1}^{\tau=t-1} \lambda^{t - \tau -1} \mathbf{h}_{\tau} \mathbf{h}_{\tau}^\intercal \right) \mathbf{h}_{t} + \mathbf{W}_x \mathbf{x}_{t}\right]\right)
    \end{aligned}
    \tag{4}\label{4}
$$
which doesn’t increase the effective depth and would be more biologically plausible.
2. **LN**: a binary variable for Layer normalization
3. **HS**: a binary variable that encodes the hidden size of the network (64, 128)

For control, we perform two more experiments dubbed **CTRL** with no fast weights and no LN, only varying the hidden size. Every experiment is repeated three times (More would not be possible for me due to limited computational budget).

The task that we choose, to perform to test our hypotheses, is associative retrieval. We start with various key-value pairs in a sequence. At the end of the sequence, one of the keys is presented and the model must predict the value that was temporarily associated with the key. Like Ba et. al. [\[1\]](#ba2016using) we used strings that contained characters from English alphabet as keys, together with the digits 0 to 9 as values.

<!-- | Input string | Target |
|:------------:|:------:|
|  c9k8j3f1??c |    9   |
|  j0a5s5z2??a |    5   | -->

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:20px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:20px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Input string</th>
    <th class="tg-7btt">Target</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">c9k8j3f1??c</td>
    <td class="tg-c3ow">9</td>
  </tr>
  <tr>
    <td class="tg-c3ow">j0a5s5z2??a</td>
    <td class="tg-c3ow">5</td>
  </tr>
</tbody>
</table>

We followed the same experimental protocol as [\[1\]](#ba2016using) and generated 100000 training examples and 10000 validation examples. Figure 2 shows the results on the validation set of all models created after all factor combinations.

{:refdef: style="text-align: center;"}
{% include image.html url="https://raw.githubusercontent.com/jeffhernandez1995/jeffhernandez1995.github.io/master/pictures/performances.svg" description="<b>Figure 2</b>: Cross entropy and accuracy of all models created after all factor combinations." %}
{: refdef}

Right from the start, we observe the following in Figure 2:
* The model with all factor combinatios **RNN-FW-LN-DEPTH-HS=128** reaches the highest accuracy basically solving the task.
* Simple RNN models, **RNN-CTRL-HS=64** and  **RNN-CTRL-HS=128** are a powerful baseline reaching median accuracies of 67% and 71%, respectively.
* The simple RNN models **RNN-CTRL-HS=64** and **RNN-CTRL-HS=128** reach superior performance than their fast weights counterparts with no increased depth **RNN-FW-HS=64** and **RNN-FW-HS=128**. Although statistical test suggest that the diference is not significant with a p-value of 0.9766 for **HS=64** and 0.8672 of **HS=128**. This would suggest that fast weights with no extra depth and a simple RNN would reach the same accuracy on this task.
* Layer normalization seems to be hurting the model **RNN-FW-LS-HS=64** but not its counterpart **RNN-FW-LS-HS=128**, which in fact reaches higher accuracies than the simple RNN baselines but this difference is still not significant with a p-value of 0.3253. This would suggest that fast weights are not in fact different from simple RNN in how the two use efficiently their weight connections.

In order to calculate effects and percentage of variance explained by the models, we use as response the increase in accuracy of the fast weights model over the average of the control models, this is $$y_{\text{FW-MODEl}} / \text{AVG}(y_{\text{CTRL-MODEl}})$$. We summarize the result of the design in Table 1.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:20px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:20px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lhti{font-style:italic;text-align:center;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-vrnj{border-color:inherit;font-style:italic;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">Factor</th>
    <th class="tg-uzvj">Effects</th>
    <th class="tg-wa1i">       %Variance</th>
    <th class="tg-wa1i">  Conf. Interval</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-vrnj">I&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">0.162</td>
    <td class="tg-nrix">---</td>
    <td class="tg-nrix">(0.116, 0.209)</td>
  </tr>
  <tr>
    <td class="tg-vrnj">LN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">0.060</td>
    <td class="tg-nrix">1.429</td>
    <td class="tg-nrix">(0.013, 0.106)</td>
  </tr>
  <tr>
    <td class="tg-lhti">DE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">0.388</td>
    <td class="tg-nrix">60.773</td>
    <td class="tg-nrix">(0.342, 0.435)</td>
  </tr>
  <tr>
    <td class="tg-lhti">HS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">0.227</td>
    <td class="tg-nrix">20.779</td>
    <td class="tg-nrix">(0.180, 0.274)</td>
  </tr>
  <tr>
    <td class="tg-lhti">LN-DE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">0.069</td>
    <td class="tg-nrix">1.923</td>
    <td class="tg-nrix">(0.022, 0.116)</td>
  </tr>
  <tr>
    <td class="tg-lhti">LN-HS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">0.103</td>
    <td class="tg-nrix">4.261</td>
    <td class="tg-nrix">(0.056, 0.150)</td>
  </tr>
  <tr>
    <td class="tg-lhti">DE-HS&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">-0.021</td>
    <td class="tg-nrix">0.184</td>
    <td class="tg-nrix">(-0.068, 0.025)*</td>
  </tr>
  <tr>
    <td class="tg-lhti">LN-DE-HS </td>
    <td class="tg-nrix">-0.059</td>
    <td class="tg-nrix">1.392</td>
    <td class="tg-nrix">(-0.106, -0.012)</td>
  </tr>
  <tr>
    <td class="tg-lhti">Error&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">---</td>
    <td class="tg-nrix">9.259</td>
    <td class="tg-nrix">---</td>
  </tr>
  <tr>
    <td class="tg-8jgo" colspan="4"><span style="font-weight:bold">Table 1: </span><span style="font-weight:normal">Effects and variance explained by the models. *=not significant</span></td>
  </tr>
</tbody>
</table>

From Table 1, we observe:
* The biggest effect is due to the increased depth, this means that as we increase this factor level from a model with no extra depth to a model with extra depth we will see an increase of in accuracy of the fast weights model over the control models. This factor accounts for 60% of the variance observed in the response, meaning that most of the observed increase in performance in this task is due to the extra depth.
* Layer normalization plays a similar role as depth, accounting for 20% of the variance observed in the response.
* Two-ways effects have a positive effect on the response, except for the combination **DEPTH-HS**, which has no significant effect.
* Factor of variation not considered (learning rate, initialization, optimizer and other hyperparameters) account for 10% of the observed variation in the response.

Finally, we can plot the response surface which offer a visual alternative to Table 1.

{:refdef: style="text-align: center;"}
{% include image.html url="https://raw.githubusercontent.com/jeffhernandez1995/jeffhernandez1995.github.io/master/notebooks/fast_weights/surface_2.svg" description="<b>Figure 3</b>: Response surface plot" %}
{: refdef}

### Fast weights: Do they really work as advertised?

Note: The title of this section is a reference to [\[5\]](#orhan2017note) which first suggested that the increased performance observed in the paper of Ba et. al. [\[1\]](#ba2016using) might actually be to factors not accounted in the authors experimental design. We note that the author of [\[5\]](#orhan2017note), unlike us, did not carry out the experiments necessary to prove this.

On this final section of our case study, I am inclined to answer that "it depends". I think that fast weights as introduced in the paper do work, they increase the performance on the task, and reach higher accuracy faster than baseline methods. Our experimental design seems to strongly indicate that this increase in performance is mainly due to the extra depth added to the model. While the author of [\[5\]](#orhan2017note) seems to suggest that the Ba et. al. have fallen in trends 1 and 2 identified in [\[6\]](#lipton2018troubling). I instead suggest that Ba et. al. only fail to identify the sources of empirical gains. Does this invalidate the results of the paper? Not at all, I consider fast weights as a type of self-attention mechanism, unlike common self-attention mechanisms which use the scaled dot-product, fast weights use the outer-product becoming a sort of "outer-product self-attention". The developments of fast weights have followed a similar fashion to its dot-product counter parts which too started with very little to no extra depth added  [\[16\]](#luong2015effective), and now extra depth is such an important part of the model that it is made explicit with extra matrices to modulate the output (these matrices are the key, query and value weights) [\[17\]](#vaswani2017attention). In this vein, we see a similar story, of incrementing fast weights with extra depth, like in [\[2\]](#zhang2017learning) where they replace equation $$\eqref{1}$$ with 

$$
    \begin{aligned}
        \mathbf{A}_t = \mathbf{W}_A \odot \mathbf{A}_{t-1} + \mathbf{W}_H \odot \mathbf{h}_t \mathbf{h}_t^\intercal,
    \end{aligned}
    \tag{5}\label{5}
$$

which according to the author allows the network to intelligently distribute inputs in $$\mathbf{A}_t$$ to increase memory capacity. Similarly, the work of [\[3\]](#schlag2018learning) which leverages several outer products to create tensors of higher orders embedded even with mode depth and capacity which as the author hypothesize introduce the combinatorial bias necessary to solve relational tasks. Finally, we note the work of [\[4\]](#le2020self) which update the original fast weights with keys, value and queries pair similar to the attention mechanism used in transformers, [\[17\]](#vaswani2017attention) but using outer-product rule instead of dot-product, which allows them to be state of the art in [Question Answering on bAbi](https://paperswithcode.com/sota/question-answering-on-babi).

## Conclusion

In this post, we have outlined the use of Design of Experiments (DoE) as tool for machine learning researches. We motivate its use with a discussion on recent troublesome trend in ML scholarship. We showed the strengths of DoE using a case study where we tested the claims of a ML paper [\[1\]](#ba2016using) using DoE and found that the observed increases in performance where due to factors not accounted in original experimental design. This methodology can be extended by including other hyper-parameters such as learning rate, optimizer, initialization schemes, random seeds, among others; and replacing the simple linear model used here with a hierarchical model as done in [\[13\]](#boquet2019decovac). The response surface methodology shown here can be used to select optimal combination of hyper-parameters in ML experiments as well as to give clarity into the true sources of empirical gains.

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

<a name="hinton1987using"></a> [\[14\]](#hinton1987using) Hinton, G. E., & Plaut, D. C. (1987, July). **Using fast weights to deblur old memories**. In Proceedings of the ninth annual conference of the Cognitive Science Society (pp. 177-186).

<a name="schmidhuber1992learning"></a> [\[15\]](#schmidhuber1992learning) Schmidhuber, J. (1992). **Learning to control fast-weight memories: An alternative to dynamic recurrent networks**. Neural Computation, 4(1), 131-139.

<a name="luong2015effective"></a> [\[16\]](#luong2015effective) Luong, M. T., Pham, H., & Manning, C. D. (2015). **Effective approaches to attention-based neural machine translation**. arXiv preprint arXiv:1508.04025.

<a name="vaswani2017attention"></a> [\[17\]](#vaswani2017attention) Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need**. In Advances in neural information processing systems (pp. 5998-6008).

{% include disqus.html %}