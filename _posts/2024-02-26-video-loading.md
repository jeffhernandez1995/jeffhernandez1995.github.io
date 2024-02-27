---
title: "Video Loading on PyTorch"
author: Jefferson Hernández
categories: pytorch, video
date:   2024-02-26 10:59:00
use_math: true
---

In this post, we will see how to load a video using PyTorch. We will various methods to load a video and convert it to a tensor. These include [VideoClips](https://github.com/pytorch/vision/blob/main/torchvision/datasets/kinetics.py) (from torchvision), [torchvision.IO](https://github.com/facebookresearch/mae_st/blob/main/util/kinetics.py) (using PyAV), and [decord](https://github.com/dmlc/decord), my own implementation using ffmpeg and [FFCV](https://github.com/libffcv/ffcv). We will use the Kinetics-400 dataset as an example. I wont tell you how to download the dataset, but you can find it [here](https://github.com/cvdfoundation/kinetics-dataset). All the code is available in this [repository](https://github.com/jeffhernandez1995/video-loading).


### Video Loading in PyTorch
We try several methods to load the Kinetics-400 dataset (241255 videos in train, 19881 videos in validations), and compare them in terms of speed and resulting size. These methods are:
- The [standard method](https://github.com/pytorch/vision/blob/main/torchvision/datasets/kinetics.py), which is to store the videos using PyTorch's VideoReader class. We use this method as a baseline. This just stores the videos as they are, mp4 files.
- Using [torchvision.IO](https://github.com/facebookresearch/mae_st/blob/main/util/kinetics.py) to load the videos. This is a library that uses the torchvision's VideoReader class and PyAV to load the videos (I was unable to make the new API work see [here](https://github.com/pytorch/vision/issues/5720)). This just requires the videos to be stored as mp4 files.
- Using [decord](https://github.com/dmlc/decord) to load the videos. This is a library that is supposed to be faster than PyTorch's VideoReader. This just requires the videos to be stored as mp4 files.
- Using the [python bindings of ffmpeg](https://github.com/kkroening/ffmpeg-python) to load the videos. This just requires the videos to be stored as mp4 files.
- Using [FFCV](https://github.com/libffcv/ffcv), we store the videos as a sequence of JPEG images. This is requires several preprocessing steps, (1) We extract the frames from the videos, (2) We resize the frames to a maximum size, (3) We encode the frames as JPEG images, (4) We store the frames in a single FFCV file. Since you are storing the frames as JPEG images, there is an increase in disk space, but the loading time is reduced. This is the method that I use in my research.

Since videos are most of the time redundat, people usually skip frames when giving them to a neural network. This is ussally denoted as $$T \times t$$, where $$T$$ is the number of frames the model sees and $$t$$ is the number of frames that are skipped in between. We use $$T=8$$ and $$t=4$$, which is the standard for Kinetics-400. Similarly, for evaluation people usually try to take full coverage of the video, so they split the video into several crops (this can be overlapin or non-overlaping) and average the predictions. Crops are denoted as $$s \times \tau$$, where $$s$$ is the number of spatial crops and $$\tau$$ is the number of time crops. We use 5 time crops an 1 spatial crop, there is no standard and this varies from paper to paper.


### Results

Datasets are loaded from RAM, and the time is measured using the `time` library. We plot the troughput in videos per second.

![troughput](https://raw.githubusercontent.com/jeffhernandez1995/jeffhernandez1995.github.io/master/pictures/throughput.png)

For the standard method you pay a fix cost prices of prepocessing the dataset of approximately 2 hours using 64 cores. But after this the datasets is very versitile you can even change the FPS of the videos. This the second fastest method, but it is the most versatile it achieves a troughput of 276.55 videos per second. The slowest method is using ffmpeg, which achieves a troughput of 26.91 videos per second. The second fastest method is using decord, which achieves a troughput of 70.75 videos per second. The third fastest method is using torchvision.IO, which achieves a troughput of 42.36 videos per second. The fastest method is using FFCV, which achieves a troughput of 564.03 videos per second. But since the FFCV approach is based on the standar method it has a higher cost of preprocessing the dataset, which is approximately 1 hour using 64 cores. The FFCV approach is the fastest, but it is the most expensive in terms of disk space, it requires 946 GB of disk space to store the training dataset and 44 GB to store the validation dataset. While all the other methods require 348 GB and 29 of disk space to store the training and validation datasets, respectively. The FFCV approach is recomended when you know that you are going to use the dataset in the same configuration several times. The standard method is recomended when you are going to use for batch inference, since it is the most versatile. The other methods are simply not recomended, since they are slower than the standard method.

### Do Kinetics-400 models generalize to other spatial and temporal views?
One question that I have always ponder is what is the best way to evaluate a video model. In image land, think ImageNet-1K, the most common way to evaluate a model is to use a single center crop. You might see people doing Multi-crop at test time techniques, but this is really not common. This is in contrast to video land, where as previously discussed this varies from paper to paper and it varies a lot. Here is a table of some examples:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">Video frames</th>
    <th class="tg-0pky">Views</th>
    <th class="tg-0pky">ViT/B-16 (84M) K400</th>
    <th class="tg-0pky">ViT/L-16 (307M) K400</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/abs/2303.12001" target="_blank">ViC-MAE</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">3x7</td>
    <td class="tg-0pky">81.5</td>
    <td class="tg-0pky">87.8</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2112.09133v2.pdf" target="_blank">MaskFeat</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">1x10</td>
    <td class="tg-0pky">82.2</td>
    <td class="tg-0pky">84.3</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2104.11227.pdf" target="_blank">MViTv1</a></td>
    <td class="tg-0pky">64x3</td>
    <td class="tg-0pky">3x3</td>
    <td class="tg-0pky">81.2</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2104.11227.pdf" target="_blank">MViTv1</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">1x5</td>
    <td class="tg-0pky">78.4</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2104.11227.pdf" target="_blank">MViTv1</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">1x10</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">80.5</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2102.05095.pdf" target="_blank">TimeSformer</a></td>
    <td class="tg-0pky">8x4</td>
    <td class="tg-0pky">3x10</td>
    <td class="tg-0pky">78</td>
    <td class="tg-0pky">80.7</td>
  </tr>
  <tr>
  <td class="tg-0pky"><a href="https://arxiv.org/pdf/2103.15691.pdf" target="_blank">ViViT</a></td>
    <td class="tg-0pky">32x2 & 32x4</td>
    <td class="tg-0pky">1x3</td>
    <td class="tg-0pky">79.2</td>
    <td class="tg-0pky">81.7</td>
  </tr>
  <tr>
  <td class="tg-0pky"><a href="https://arxiv.org/pdf/2201.04288v4.pdf" target="_blank">MTV</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">3x4</td>
    <td class="tg-0pky">81.8</td>
    <td class="tg-0pky">84.3</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2106.05392v2.pdf" target="_blank">Mformer</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">3x10</td>
    <td class="tg-0pky">79.7</td>
    <td class="tg-0pky">80.2</td>
  </tr>
  <tr>
  <td class="tg-0pky"><a href="https://arxiv.org/pdf/2104.14294.pdf" target="_blank">DINOV1</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">3x10</td>
    <td class="tg-0pky">82.5</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2303.16058v1.pdf" target="_blank">UMT</a></td>
    <td class="tg-0pky">8x2</td>
    <td class="tg-0pky">4x3</td>
    <td class="tg-0pky">85.7</td>
    <td class="tg-0pky">90.6</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2212.04500v2.pdf" target="_blank">MVD</a></td>
    <td class="tg-0pky">16x4</td>
    <td class="tg-0pky">1x10</td>
    <td class="tg-0pky">83.4</td>
    <td class="tg-0pky">87.2</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2211.09552.pdf" target="_blank">UniFormerV2</a></td>
    <td class="tg-0pky">8x2</td>
    <td class="tg-0pky">3x4</td>
    <td class="tg-0pky">85.6</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/2211.09552.pdf" target="_blank">UniFormerV2</a></td>
    <td class="tg-0pky">32x8</td>
    <td class="tg-0pky">2x3</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">89.3</td>
  </tr>

  <tr>
    <td class="tg-8jgo" colspan="5"><span style="font-weight:bold">Table 1: </span><span style="font-weight:normal">Examples of video model evaluation on the Kinetics-400 benchmark.</span></td>
  </tr>
</tbody>
</table>

As we can see numbers are all over the place, and there is no standard.  Look at the table and tell me what model is better? It is very hard to tell. Let's do an exercise and evaluate some open souce models on Hugging Face using the FFCV approach. But first lets us think what would be a fair evaluation? I think this is not a question with an easy answer, and the current consensus seems to be _do what shows best perfomance on the benchmarks_. I am guilty of falling into this myself with the ViC-MAE results. 

Following similar steps to the image community, indicates that the best most fair would be a center spatial and a center temporal view. But this is a very strong assumption videos are more messy than images, assuming that the action occurs at the center of the video is a very strong assumption, even stronger (to me) than the action occuring at the center of the frame (like in datasets like ImageNet-1K, where given how they where collected this is a reasonable assumption). I am gonna go ahead an use 1 center crop and 5 center temporal crops. There might be a better way to do this, but I think this is a good start. My reasoning stems that if most models are trained using 16 frames skiping 4 (or 32 skiping 2), you would only need 5 temporal crops to cover the whole video.

The models on the HUB that I found are:
- [VideoMAE](https://arxiv.org/abs/2203.12602) on the following configurations: [Small](https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics), [Base](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics), [Large](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics), [Huge](https://huggingface.co/MCG-NJU/videomae-huge-finetuned-kinetics). Trained using 16 frames and skipping 4.
- My own [ViC-MAE](https://arxiv.org/abs/2303.12001) on the following configurations: Base, Large. See the ViC-MAE [repository](https://github.com/jeffhernandez1995/ViC-MAE) for more details. Trained using 16 frames and skipping 4.
- [TimeSformer](https://arxiv.org/pdf/2102.05095.pdf) on the following configurations: [Base](https://huggingface.co/facebook/timesformer-base-finetuned-k400), [HR](https://huggingface.co/facebook/timesformer-hr-finetuned-k400). Trained using 8 frames and skipping 4 and 16 frames and skipping 4, respectively.
- [ViViT](https://arxiv.org/pdf/2103.15691.pdf) on the following configurations: [Base](https://huggingface.co/google/vivit-b-16x2-kinetics400). Trained using 32 frames and skipping 2.

You can see the results in table format [here](https://github.com/jeffhernandez1995/video-loading).
![results](https://raw.githubusercontent.com/jeffhernandez1995/jeffhernandez1995.github.io/master/pictures/results.png)

We see a significant drop in accuracy, on average approximately 3%. If you think about it on 2022 the year VideoMAE came out  the best model was [MVT/H](https://arxiv.org/abs/2201.04288v4) with an accuracy of 89.9


{% include disqus.html %}