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

"""Tests for encoder modules."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import numpy as np
from videoprism import encoders


class EncodersTest(parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test_embedding_layer(self):
    num_classes, dim, input_shape = 8, 10, (5, 20)
    npy_input = np.random.randint(0, num_classes, input_shape).astype('int32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    emb_layer = encoders.Embedding(
        name='emb_lookup',
        num_classes=num_classes,
        input_dim=dim,
        scale_sqrt_depth=True,
    )

    @self.variant
    def var_fn():
      return emb_layer.init_with_output(prng_key, inputs)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 1)
    self.assertEqual(outputs.shape, input_shape + (dim,))

  @chex.variants(with_jit=True, without_jit=True)
  def test_positional_embedding_layer(self):
    seq_len, dim = 8, 10
    prng_key = jax.random.PRNGKey(seed=123)
    emb_layer = encoders.PositionalEmbedding(
        name='pos_emb',
        embedding_dim=dim,
        min_timescale=10,
        max_timescale=20,
    )

    @self.variant
    def var_fn():
      return emb_layer.init_with_output(prng_key, seq_len)

    outputs, params = var_fn()
    self.assertEmpty(jax.tree_util.tree_flatten(params)[0])
    self.assertEqual(outputs.shape, (1, seq_len, dim))

  @chex.variants(with_jit=True, without_jit=True)
  def test_trainable_positional_embedding_layer(self):
    seq_len, dim = 8, 10
    prng_key = jax.random.PRNGKey(seed=123)
    emb_layer = encoders.TrainablePositionalEmbedding(
        name='pos_emb',
        max_seq_length=seq_len,
        embedding_dim=dim,
        lookup_style='matmul',
    )

    @self.variant
    def var_fn():
      return emb_layer.init_with_output(prng_key, seq_len)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 1)
    self.assertEqual(outputs.shape, (1, seq_len, dim))

  @chex.variants(with_jit=True)
  @parameterized.product(
      scan=[True, False],
      train=[True, False],
  )
  def test_vision_transformer(self, scan: bool, train: bool):
    batch_size, seq_len, dim = 1, 6, 4
    np_inputs = np.random.normal(1.0, 0.5, [batch_size, seq_len, dim]).astype(
        'float32'
    )
    inputs = jnp.asarray(np_inputs)
    prng_key = jax.random.PRNGKey(seed=123)
    vit = encoders.VisionTransformer(
        name='vit',
        num_tfm_layers=2,
        mlp_dim=4,
        num_heads=2,
        scan=scan,
    )

    @self.variant
    def var_fn():
      return vit.init_with_output(prng_key, inputs, train=train)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 16 if scan else 32)
    self.assertEqual(outputs.shape, (batch_size, seq_len, dim))

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      ('train', False, True, False, False),
      ('scan', True, False, False, False),
      ('scan_and_train', True, True, False, False),
      ('return_intermediate', True, False, True, False),
      ('use_frame_paddings', True, False, False, True),
  )
  def test_factorized_encoder(
      self,
      scan: bool,
      train: bool,
      return_intermediate: bool,
      use_frame_paddings: bool,
  ):
    batch_size, num_frames, image_size, patch_size, dim = 1, 4, 16, 4, 8
    np_inputs = np.random.normal(
        0.0,
        0.1,
        [batch_size, num_frames, image_size, image_size, 3],
    ).astype('float32')
    inputs = jnp.asarray(np_inputs)

    frame_paddings = None
    if use_frame_paddings:
      np_frame_paddings = np.zeros((batch_size, num_frames), dtype='float32')
      np_frame_paddings[:, num_frames // 2 :] = 1
      frame_paddings = jnp.asarray(np_frame_paddings)

    prng_key = jax.random.PRNGKey(seed=123)
    enc = encoders.FactorizedEncoder(
        name='enc',
        patch_size=patch_size,
        pos_emb_shape=(16, 16, 16),
        model_dim=dim,
        num_spatial_layers=2,
        num_temporal_layers=2,
        num_heads=2,
        mlp_dim=4,
        atten_logit_cap=50.0,
        scan=scan,
    )

    @self.variant
    def var_fn():
      return enc.init_with_output(
          prng_key,
          inputs,
          train=train,
          return_intermediate=return_intermediate,
          frame_paddings=frame_paddings,
      )

    (embeddings, outputs), params = var_fn()

    self.assertLen(jax.tree_util.tree_flatten(params)[0], 40 if scan else 72)
    self.assertEqual(
        embeddings.shape,
        (batch_size, num_frames * (image_size // patch_size) ** 2, dim),
    )
    if return_intermediate:
      self.assertEqual(
          outputs['spatial_features'].shape,
          (batch_size, num_frames * (image_size // patch_size) ** 2, dim),
      )
    else:
      self.assertEmpty(outputs)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      ('train', False, True),
      ('scan', True, False),
      ('scan_and_train', True, True),
  )
  def test_text_encoder(self, scan: bool, train: bool):
    batch_size, seq_len, vocab_size, dim = 1, 10, 20, 8
    np_inputs = np.random.randint(0, vocab_size, [batch_size, seq_len]).astype(
        'int32'
    )
    inputs = jnp.asarray(np_inputs)
    np_paddings = np.zeros([batch_size, seq_len], dtype='float32')
    np_paddings[:, seq_len // 2 :] = 1
    paddings = jnp.asarray(np_paddings)

    prng_key = jax.random.PRNGKey(seed=123)
    enc = encoders.TextEncoder(
        name='enc',
        vocabulary_size=vocab_size,
        num_class_tokens=1,
        model_dim=dim,
        num_layers=2,
        mlp_dim=4,
        num_heads=2,
        atten_logit_cap=50.0,
        scan=scan,
    )

    @self.variant
    def var_fn():
      return enc.init_with_output(prng_key, inputs, paddings, train=train)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 20 if scan else 36)
    self.assertEqual(outputs.shape, (batch_size, seq_len + 1, dim))

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      ('train', False, True, False),
      ('scan_and_train', True, True, False),
      ('return_intermediate', True, False, True),
  )
  def test_factorized_video_clip(
      self, scan: bool, train: bool, return_intermediate: bool
  ):
    batch_size, num_frames, image_size, patch_size, dim = 1, 4, 16, 4, 8
    np_inputs = np.random.normal(
        0.0,
        0.1,
        [batch_size, num_frames, image_size, image_size, 3],
    ).astype('float32')
    inputs = jnp.asarray(np_inputs)

    seq_len, vocab_size = 10, 20
    np_texts = np.random.randint(
        0, vocab_size, [batch_size, seq_len]
    ).astype('int32')
    texts = jnp.asarray(np_texts)
    np_text_paddings = np.zeros([batch_size, seq_len], dtype='float32')
    np_text_paddings[:, seq_len // 2 :] = 1
    text_paddings = jnp.asarray(np_text_paddings)

    prng_key = jax.random.PRNGKey(seed=123)
    net = encoders.FactorizedVideoCLIP(
        name='net',
        patch_size=patch_size,
        pos_emb_shape=(16, 16, 16),
        num_spatial_layers=2,
        num_temporal_layers=2,
        mlp_dim=4,
        num_auxiliary_layers=1,
        vocabulary_size=vocab_size,
        enable_causal_atten=True,
        num_unimodal_layers=2,
        norm_policy='pre',
        model_dim=dim,
        num_heads=2,
        atten_logit_cap=50.0,
        scan=scan,
    )

    @self.variant
    def var_fn():
      return net.init_with_output(
          prng_key,
          inputs=inputs,
          texts=texts,
          text_paddings=text_paddings,
          train=train,
          normalize=True,
          return_intermediate=return_intermediate,
      )

    (vision_embeddings, text_embeddings, outputs), params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 88 if scan else 136)
    self.assertEqual(vision_embeddings.shape, (batch_size, dim))
    self.assertEqual(text_embeddings.shape, (batch_size, dim))
    if return_intermediate:
      self.assertEqual(
          outputs['spatial_features'].shape,
          (batch_size, num_frames * (image_size // patch_size) ** 2, dim),
      )
      self.assertEqual(
          outputs['spatiotemporal_features'].shape,
          (batch_size, num_frames * (image_size // patch_size) ** 2, dim),
      )
    else:
      self.assertEmpty(outputs)


if __name__ == '__main__':
  absltest.main()
