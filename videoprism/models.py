# Copyright 2025 VideoPrism Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides builders and loaders of VideoPrism checkpoints.

The v1 base model takes videos with shape (16, 288, 288) as inputs and outputs
embeddings with shape (batch_size, 4096, 768) which could be reshaped into
(batch_size, 16, 16, 16, 768) for spatiotemporal representations. The input
videos should be normalized in [0.0, 1.0].

Example usage:
```
from videoprism import models as vp

model_name = 'videoprism_public_v1_base'
flax_model = vp.MODELS[model_name]()
loaded_state = vp.load_pretrained_weights(model_name)

@jax.jit
def forward_fn(inputs):
  return flax_model.apply(loaded_state, inputs, train=False)

model_inputs = ...
outputs = forward_fn(model_inputs)
```
"""

from collections.abc import Mapping, Sequence

import flax
import huggingface_hub
import numpy as np
from videoprism import encoders
from videoprism import tokenizers
from videoprism import utils

TEXT_TOKENIZERS = {
    'c4_en': (  # vocab_size=32_000
        'gs://t5-data/vocabs/cc_en.32000/sentencepiece.model'
    ),
}

CHECKPOINTS = {
    'videoprism_public_v1_base': (
        'gs://videoprism/v1/flax_base_f16r288_repeated.npz'
    ),
    'videoprism_public_v1_large': (
        'gs://videoprism/v1/flax_large_f8r288_repeated.npz'
    ),
    # Hugging Face checkpoints (repository, filename)
    'videoprism_public_v1_base_hf': (
        'google/videoprism-base-f16r288',
        'flax_base_f16r288_repeated.npz',
    ),
    'videoprism_public_v1_large_hf': (
        'google/videoprism-large-f8r288',
        'flax_large_f8r288_repeated.npz',
    ),
    'google/videoprism-base-f16r288': (
        'google/videoprism-base-f16r288',
        'flax_base_f16r288_repeated.npz',
    ),
    'google/videoprism-large-f8r288': (
        'google/videoprism-large-f8r288',
        'flax_large_f8r288_repeated.npz',
    ),
}

CONFIGS = {
    'videoprism_v1_base': dict(
        patch_size=18,
        pos_emb_shape=(16, 16, 16),
        model_dim=768,
        num_spatial_layers=12,
        num_temporal_layers=4,
        num_heads=12,
        mlp_dim=3072,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_v1_large': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        model_dim=1024,
        num_spatial_layers=24,
        num_temporal_layers=4,
        num_heads=16,
        mlp_dim=4096,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_v1_giant': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        model_dim=1408,
        num_spatial_layers=40,
        num_temporal_layers=4,
        num_heads=16,
        mlp_dim=6144,
        atten_logit_cap=50.0,
        scan=True,
    ),
}


def videoprism_v1_base():
  """Builds VideoPrism v1 base model."""
  return encoders.FactorizedEncoder(**CONFIGS['videoprism_v1_base'])


def videoprism_v1_large():
  """Builds VideoPrism v1 large model."""
  return encoders.FactorizedEncoder(**CONFIGS['videoprism_v1_large'])


def videoprism_v1_giant():
  """Builds VideoPrism v1 giant model."""
  return encoders.FactorizedEncoder(**CONFIGS['videoprism_v1_giant'])


MODELS = {
    'videoprism_public_v1_base': videoprism_v1_base,
    'videoprism_public_v1_large': videoprism_v1_large,
    'videoprism_public_v1_base_hf': videoprism_v1_base,
    'videoprism_public_v1_large_hf': videoprism_v1_large,
}


def load_pretrained_weights(
    model_name: str | None,
    checkpoint_path: str | None = None,
    checkpoints: Mapping[str, str] | None = None,
):
  """Loads pretrained model weight from Google Cloud storage or HuggingFace.

  Args:
    model_name: A string for the model name.
    checkpoint_path: Optional path of the model checkpoint.
    checkpoints: Mapping from model name to checkpoint path. Used with
      `model_name`. If None, use the default `CHECKPOINTS`.

  Returns:
    Restored Flax model weights.
  """
  checkpoints = checkpoints or CHECKPOINTS
  checkpoint_path = checkpoint_path or checkpoints.get(model_name)
  # Handle Hugging Face checkpoints using suffix _hf.
  if model_name is not None and (
      model_name.startswith('google/') or model_name.endswith('_hf')
  ):
    repo_id, filename = checkpoint_path
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=filename
    )
  variables = utils.load_checkpoint(checkpoint_path)
  return flax.core.freeze(variables)


def load_text_tokenizer(name: str) -> tokenizers.Tokenizer:
  """Loads a text tokenizer by name.

  Args:
    name: A string for the text tokenizer model name.

  Returns:
    A text tokenizer.
  """
  if name not in TEXT_TOKENIZERS:
    raise ValueError(f'Text tokenizer `{name}` not found.')

  model_path = TEXT_TOKENIZERS[name]
  return tokenizers.SentencePieceTokenizer(model_path)


def tokenize_texts(
    tokenizer: tokenizers.Tokenizer,
    inputs: Sequence[str],
    max_length: int,
    add_bos: bool | None = None,
    canonicalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
  """Tokenizes a batch of texts.

  Args:
    tokenizer: The tokenizer to use.
    inputs: The list of texts to tokenize.
    max_length: The maximum length of the tokenized texts.
    add_bos: Whether to add a beginning-of-sentence token. If None, the
      beginning-of-sentence token will be added if the tokenizer's bos_token is
      a non-negative integer.
    canonicalize: Whether to canonicalize the texts before tokenization.

  Returns:
    A tuple of two numpy arrays containing the padded token ids and the
    corresponding paddings, where 1 denotes padding token.
  """

  if canonicalize:
    inputs = [utils.canonicalize_text(text) for text in inputs]

  batch_ids, batch_paddings = [], []
  if add_bos is None:
    add_bos = tokenizer.bos_token >= 0

  for ids in tokenizer.to_int(inputs, bos=add_bos, eos=False):
    ids_seq_len = len(ids)
    if ids_seq_len > max_length:
      ids = ids[:max_length]

    ids = np.asarray(ids, dtype=np.int32)
    paddings = np.zeros_like(ids, dtype=np.float32)

    if ids_seq_len < max_length:
      ids = np.pad(
          ids, (0, max_length - ids_seq_len), 'constant', constant_values=0
      )
      paddings = np.pad(
          paddings,
          (0, max_length - ids_seq_len),
          'constant',
          constant_values=1.0,
      )

    batch_ids.append(ids)
    batch_paddings.append(paddings)

  return np.stack(batch_ids), np.stack(batch_paddings)
