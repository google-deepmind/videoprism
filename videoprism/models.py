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

from collections.abc import Mapping

import flax
from videoprism import encoders
from videoprism import utils

TEXT_MAX_LEN: int = 64
TEXT_VOCAB_SIZE: int = 64_000
TEXT_SPM_PATH: str = 'gs://videoprism/vocabs/64000/sentencepiece.model'

CHECKPOINTS = {
    'videoprism_public_v1_base': (
        'gs://videoprism/v1/flax_base_f16r288_repeated.npz'
    ),
    'videoprism_public_v1_large': (
        'gs://videoprism/v1/flax_large_f8r288_repeated.npz'
    ),
    'videoprism_lvt_public_v1_base': (
        'gs://videoprism/v1/flax_lvt_base_f16r288_repeated.npz'
    ),
    'videoprism_lvt_public_v1_large': (
        'gs://videoprism/v1/flax_lvt_large_f8r288_repeated.npz'
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
    'videoprism_lvt_v1_base': dict(
        patch_size=18,
        pos_emb_shape=(16, 16, 16),
        num_spatial_layers=12,
        num_temporal_layers=4,
        mlp_dim=3072,
        num_auxiliary_layers=2,
        vocabulary_size=TEXT_VOCAB_SIZE,
        enable_causal_atten=True,
        num_unimodal_layers=12,
        norm_policy='pre',
        model_dim=768,
        num_heads=12,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_lvt_v1_large': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        num_spatial_layers=24,
        num_temporal_layers=4,
        mlp_dim=4096,
        num_auxiliary_layers=2,
        vocabulary_size=TEXT_VOCAB_SIZE,
        enable_causal_atten=True,
        num_unimodal_layers=12,
        norm_policy='pre',
        model_dim=1024,
        num_heads=16,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_lvt_v1_giant': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        num_spatial_layers=40,
        num_temporal_layers=4,
        mlp_dim=6144,
        num_auxiliary_layers=2,
        vocabulary_size=TEXT_VOCAB_SIZE,
        enable_causal_atten=True,
        num_unimodal_layers=16,
        norm_policy='primer_hybrid',
        model_dim=1408,
        num_heads=16,
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


def videoprism_lvt_v1_base():
  """Builds VideoPrism LvT v1 base model."""
  return encoders.FactorizedVideoCLIP(**CONFIGS['videoprism_lvt_v1_base'])


def videoprism_lvt_v1_large():
  """Builds VideoPrism LvT v1 large model."""
  return encoders.FactorizedVideoCLIP(**CONFIGS['videoprism_lvt_v1_large'])


def videoprism_lvt_v1_giant():
  """Builds VideoPrism LvT v1 giant model."""
  return encoders.FactorizedVideoCLIP(**CONFIGS['videoprism_lvt_v1_giant'])


MODELS = {
    'videoprism_public_v1_base': videoprism_v1_base,
    'videoprism_public_v1_large': videoprism_v1_large,
    'videoprism_lvt_public_v1_base': videoprism_lvt_v1_base,
    'videoprism_lvt_public_v1_large': videoprism_lvt_v1_large,
}


def load_pretrained_weights(
    model_name: str | None,
    checkpoint_path: str | None = None,
    checkpoints: Mapping[str, str] | None = None,
):
  """Loads pretrained model weight.

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
  variables = utils.load_checkpoint(checkpoint_path)
  return flax.core.freeze(variables)
