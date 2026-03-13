# VideoPrism: A Foundational Visual Encoder for Video Understanding

[![Paper](https://img.shields.io/badge/arXiv-2402.13217-red.svg)](https://arxiv.org/abs/2402.13217)
[![Blog](https://img.shields.io/badge/Google_Research-Blog-green.svg)](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/)
[![Video Encoder Colab Demo](https://img.shields.io/static/v1?label=Video%20Encoder%20Demo&message=Google%20Colab&logo=google&color=orange)](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_encoder_demo.ipynb)
[![Video-Text Encoder Colab Demo](https://img.shields.io/static/v1?label=Video-Text%20Encoder%20Demo&message=Google%20Colab&logo=google&color=orange)](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_text_demo.ipynb)
[![Video Classification Colab Demo](https://img.shields.io/static/v1?label=Video%20Classification%20Demo&message=Google%20Colab&logo=google&color=orange)](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_classification_demo.ipynb)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/google/videoprism-686e823d6070ec6ad9e4b1f2)
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

## Updates

* **[Mar-13-26]:** Added video classification fine-tuning with the frozen backbone [[`Colab notebook`](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_encoder_demo.ipynb)]. :fire::fire:
* **[Jul-16-25]:** Released VideoPrism video-text encoders for cross-modal retrieval [[`Colab notebook`](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_text_demo.ipynb)]. :fire::fire:
* **[Jun-15-25]:** Added models to [[`Hugging Face`](https://huggingface.co/collections/google/videoprism-686e823d6070ec6ad9e4b1f2)].
* **[Jun-05-25]:** Added video encoder demo [[`Colab notebook`](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_encoder_demo.ipynb)].
* **[Jun-03-25]:** Released VideoPrism video encoders (Base and Large) [[`Blog`](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/)] [[`Paper`](https://arxiv.org/abs/2402.13217)]. :fire::fire:

## TODOs

- [ ] Add PyTorch model support.

## Getting started

You will need Python 3.9 or later. Download the code from GitHub and run:

```shell
$ git clone https://github.com/google-deepmind/videoprism.git
$ cd videoprism
$ pip install .
```

Please get started with the following example code for model checkpoint loading
and inference or use the [Colab notebook for video encoders](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_encoder_demo.ipynb) / [Colab notebook for video-text encoders](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_text_demo.ipynb):

```python
import jax
from videoprism import models as vp

# Video encoders.
model_name = 'videoprism_public_v1_base'  # configuration name
flax_model = vp.get_model(model_name)
loaded_state = vp.load_pretrained_weights(model_name)

@jax.jit
def forward_fn(inputs):
  return flax_model.apply(loaded_state, inputs, train=False)

video_inputs = ...  # Shape = [batch_size, num_frames, height, width, 3].
outputs, _ = forward_fn(video_inputs)  # Shape = [batch_size, num_tokens, feature_channels].

# Video-text encoders.
model_name = 'videoprism_lvt_public_v1_base'  # configuration name
flax_model = vp.get_model(model_name)
loaded_state = vp.load_pretrained_weights(model_name)
text_tokenizer = vp.load_text_tokenizer('c4_en')

@jax.jit
def forward_fn(inputs, text_token_ids, text_token_paddings, train=False):
  return flax_model.apply(
      loaded_state,
      inputs,
      text_token_ids,
      text_token_paddings,
      train=train,
  )

video_inputs = ...  # Shape = [batch_size, num_frames, height, width, 3].
text_queries = ...  # A list of input text queries.
text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)
video_embeddings, text_embeddings, _ = forward_fn(
  video_inputs, text_ids, text_paddings)  # Shape = [batch_size, feature_channels].
```

## Video Classification example
We provide a [Colab notebook for video classification](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_classification_demo.ipynb) to
show how to fine-tune VideoPrism for video classification by keeping
the pre-trained backbone frozen and training only
a lightweight attention-pooler + projection head.

## Released models

We release the following model variants:

| Model Name | Configuration Name | Model Type | Backbone | #Params | File Size | Checkpoint |
| -------- | -------- | ------- | :-------: | :-------: | :-------: | :-------: |
| VideoPrism-B | `videoprism_public_v1_base`  | Video encoder | ViT-B | 114M | 458MB | [link](https://huggingface.co/google/videoprism-base-f16r288) |
| VideoPrism-L | `videoprism_public_v1_large` | Video encoder | ViT-L | 354M | 1.42GB | [link](https://huggingface.co/google/videoprism-large-f8r288) |
| VideoPrism-LvT-B | `videoprism_lvt_public_v1_base`  | Video-text encoders | ViT-B | 248M | 991MB | [link](https://huggingface.co/google/videoprism-lvt-base-f16r288) |
| VideoPrism-LvT-L | `videoprism_lvt_public_v1_large` | Video-text encoders | ViT-L | 580M | 2.30GB | [link](https://huggingface.co/google/videoprism-lvt-large-f8r288) |

Video encoders take videos with shape `(batch_size, num_frames, 288, 288, 3)`
as inputs and output embeddings with shape
`(batch_size, num_frames * 16 * 16, feature_channels)` which could be reshaped
into `(batch_size, num_frames, 16, 16, feature_channels)` for spatiotemporal
representations. During model training, `num_frames` is set to 16 and 8 for
VideoPrism-B and VideoPrism-L, respectively. Both models are expected to work
with arbitrary `num_frames` by interpolating the temporal positional embeddings.
The RGB values of input videos should be normalized in [0.0, 1.0].

In video-text models, both video and text encoders produce global embeddings
with shape `(batch_size, feature_channels)`, whose similarities could be
measured by cosine distances. We use the `c4_en` [SentencePiece](https://github.com/google/sentencepiece) model for text tokenization. During inference, embedding
calculation for either modality can be skipped by providing `None` as the input.

## Results with frozen backbones

*"Public"* denotes models we released in this repository. *"Paper"* and
*"Prior SOTA"* denote our models and previous best-performing models reported
in the [paper](https://arxiv.org/abs/2402.13217), respectively. Our *public*
models perform slightly worse than the *paper* models due to different
pre-training image-text data we used subject to data policy.

### Video-focused tasks ([VideoGLUE](https://arxiv.org/abs/2307.03166))

| Models | K400 | MiT | SSv2 | D48 | Charades | ActivityNet | AVA | AVA-K |
| -------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **VideoPrism-B (public)** | 82.9 | 39.7 | 62.2 | 64.3 | 43.5 | 36.5 | 28.3 | 30.8 |
| **VideoPrism-L (public)** | 85.0 | 43.3 | 64.6 | 67.6 | 53.2 | 37.0 | 32.4 | 34.5 |
| VideoPrism-B (paper) | 84.2 | 40.8 | 63.6 | 67.4 | 40.4 | 36.6 | 30.6 | 31.8 |
| VideoPrism-g (paper) | 87.2 | 45.5 | 68.5 | 71.3  | 62.3 | 37.8 | 36.2 | 37.3 |
| Prior SOTA (B) | 77.1 | 34.0 | 58.2 | 55.6 | 33.3 | 35.8 | 21.1 | 25.9 |
| Prior SOTA (L+) | 82.8 | 40.3 | 67.4 | 69.6 | 39.9 | 36.7 | 24.4 | 26.2 |

### Zero-shot video-text retrieval

| Models | MSRVTT-1K (v2t)  | MSRVTT-1K (t2v) | VATEX (v2t) | VATEX (t2v) | ActivityNet (v2t) | ActivityNet (t2v) |
| -------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **VideoPrism-LvT-B (public)** | 49.8 | 50.1 | 73.1 | 56.2 | 47.9 | 48.8 |
| **VideoPrism-LvT-L (public)** | 50.6 | 50.1 | 75.0 | 57.2 | 49.1 | 51.3 |
| VideoPrism-LvT-B (paper) | 50.2 | 51.4 | 76.2 | 57.7 | 47.9 | 49.6 |
| VideoPrism-LvT-g (paper) | 51.7 | 52.7 | 77.1 | 62.5 | 50.3 | 52.7 |
| Prior SOTA (B) | - | 34.0 | - | - | - | 30.6 |
| Prior SOTA (L+) | 45.4 | 43.9 | 73.6 | 53.2 | 40.7 | 42.8 |

### Zero-shot video classification

| Models | K400 | SSv2 (Temporal) | SSv2 (Events) | NExT-QA (Hard) | Charades | Charades (STA) |
| -------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **VideoPrism-LvT-B (public)** | 69.2 | 14.6 | 11.3 | 31.1 | 26.9 | 48.6 |
| **VideoPrism-LvT-L (public)** | 72.4 | 18.0 | 12.4 | 32.1 | 32.4 | 50.2 |
| VideoPrism-LvT-B (paper) | 71.3 | 16.1 | 11.9 | 31.3 | 29.2 | 50.0 |
| VideoPrism-LvT-g (paper) | 74.6 | 18.6 | 15.7 | 32.7 | 32.4 | 50.4 |
| Prior SOTA (B) | - | 9.8 | 6.4 | 27.6 | 21.1 | - |
| Prior SOTA (L+) | 72.0 | 15.2 | 11.4 | 25.2 | 25.8 | 47.2 |

## Citation

If you use VideoPrism, please cite the following papers:

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
```bibtex
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

@article{sun2024video,
  title={Video foundation models for animal behavior analysis},
  author={Sun, Jennifer J and Zhou, Hao and Yang, Xinyu and Zhao, Long and Yuan, Liangzhe and Seybold, Bryan and Hendon, David and Schroff, Florian and Ross, David A and Adam, Hartwig and Hu, Bo and Liu, Ting},
  journal={bioRxiv},
  pages={2024--07},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
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