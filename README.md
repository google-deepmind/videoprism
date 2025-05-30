# VideoPrism: A Foundational Visual Encoder for Video Understanding

[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2402.13217)
[![Blog](https://img.shields.io/badge/Google_Research-Blog-green.svg)](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[VideoPrism](https://arxiv.org/abs/2402.13217) is a general-purpose video
encoder designed to handle a wide spectrum of video understanding tasks,
including classification, retrieval, localization, captioning, and question
answering. It is pre-trained on a massive and diverse dataset: 1 billion
image-text pairs from [WebLI](https://arxiv.org/abs/2209.06794), 36 million
high-quality video-text pairs, and 582 million video clips with noisy or
machine-generated parallel text (subject to data wipeout). The pre-training
approach is designed for these hybrid data, to learn both from video-text pairs
and the videos themselves. VideoPrism is fairly easy to adapt to new video
understanding tasks, and achieves state-of-the-art performance on 31 out of 33
public video understanding benchmarks using a single frozen model.

This repository releases the model weight checkpoints and hosts [JAX](https://github.com/jax-ml/jax)/[Flax](https://github.com/google/flax) utility
functions for checkpoint loading and model inference.

## Getting started

You will need Python 3.9 or later. Download the code from GitHub and run:

```shell
$ git clone https://github.com/google-deepmind/videoprism.git
$ cd videoprism
$ pip install .
```

Below shows example code for model checkpoint loading and inference:

```python
import jax
from videoprism import models as vp

model_name = 'videoprism_public_v1_large'  # configuration name
flax_model = vp.MODELS[model_name]()
loaded_state = vp.load_pretrained_weights(model_name)

@jax.jit
def forward_fn(inputs):
  return flax_model.apply(loaded_state, inputs, train=False)

model_inputs = ...  # Shape = [batch_size, num_frames, height, width, 3].
outputs = forward_fn(model_inputs)  # Shape = [batch_size, num_tokens, feature_channels].
```

**Note:** Please make sure that the model `apply` function is wrapped in
`jax.jit` to get the correct results.

## Released models

We release the following model variants:

| Model Name | Configuration Name | Model Type | Backbone | #Params |
| -------- | -------- | ------- | :-------: | :-------: |
| VideoPrism-B | `videoprism_public_v1_base`  | Video encoder | ViT-B | 114M |
| VideoPrism-L | `videoprism_public_v1_large` | Video encoder | ViT-L | 354M |

The models take videos with shape `(num_frames, 288, 288, 3)` as inputs and
outputs embeddings with shape `(num_frames * 16 * 16, feature_channels)` which
could be reshaped into `(num_frames, 16, 16, feature_channels)` for
spatiotemporal representations. During model training, `num_frames` is set to 16
and 8 for VideoPrism-B and VideoPrism-L, respectively. Both models are expected
to work with arbitrary `num_frames` by interpolating the temporal positional
embeddings. The RGB values of input videos should be normalized in [0.0, 1.0].

### Results on video-focused tasks ([VideoGLUE](https://arxiv.org/abs/2307.03166)) with frozen backbones

| Dataset | K400 | MiT | SSv2 | D48 | Charades | ActivityNet | AVA | AVA-K |
| -------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **VideoPrism-B (public)** | 82.9 | 39.7 | 62.2 | 64.3 | 43.5 | 36.5 | 28.3 | 30.8 |
| **VideoPrism-L (public)** | 85.0 | 43.3 | 64.6 | 67.6 | 53.2 | 37.0 | 32.4 | 34.5 |
| VideoPrism-B (paper) | 84.2 | 40.8 | 63.6 | 67.4 | 40.4 | 36.6 | 30.6 | 31.8 |
| VideoPrism-g (paper) | 87.2 | 45.5 | 68.5 | 71.3  | 62.3 | 37.8 | 36.2 | 37.3 |
| Prior SOTA (B) | 77.1 | 34.0 | 58.2 | 55.6 | 33.3 | 35.8 | 21.1 | 25.9 |
| Prior SOTA (L+) | 82.8 | 40.3 | 67.4 | 69.6 | 39.9 | 36.7 | 24.4 | 26.2 |

### Results on zero-shot video-text retrieval and classification tasks with frozen backbones

*"Public"* denotes models we released in this repository. *"Paper"* and
*"Prior SOTA"* denote our models and previous best-performing models reported
in the [paper](https://arxiv.org/abs/2402.13217), respectively. Our *public*
models perform slightly worse than the *paper* models due to different
pre-training image-text data we used subject to data policy.

## Citation

If you use VideoPrism, please cite the following papers:

```
@inproceedings{zhao2024videoprism,
  title = {{VideoPrism}: A Foundational Visual Encoder for Video Understanding},
  author = {Long Zhao and Nitesh B. Gundavarapu and Liangzhe Yuan and Hao Zhou and Shen Yan and Jennifer J. Sun and Luke Friedman and Rui Qian and Tobias Weyand and Yue Zhao and Rachel Hornung and Florian Schroff and Ming-Hsuan Yang and David A. Ross and Huisheng Wang and Hartwig Adam and Mikhail Sirotenko and Ting Liu and Boqing Gong},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2024}
}

@article{yuan2024videoglue,
  title = {{VideoGLUE}: Video General Understanding Evaluation of Foundation Models},
  author = {Liangzhe Yuan and Nitesh Bharadwaj Gundavarapu and Long Zhao and Hao Zhou and Yin Cui and Lu Jiang and Xuan Yang and Menglin Jia and Tobias Weyand and Luke Friedman and Mikhail Sirotenko and Huisheng Wang and Florian Schroff and Hartwig Adam and Ming-Hsuan Yang and Ting Liu and Boqing Gong},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year = {2024}
}
```

## License

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license. You
may obtain a copy of the Apache 2.0 license at: <https://www.apache.org/licenses/LICENSE-2.0>

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: <https://creativecommons.org/licenses/by/4.0/legalcode>

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

## Disclaimer

This is not an official Google product.