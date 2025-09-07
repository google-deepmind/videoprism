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

"""Tests for layer modules."""

from absl.testing import absltest
from absl.testing import parameterized
import unittest
try:
  import chex  # type: ignore
  import jax  # type: ignore
  from jax import numpy as jnp  # type: ignore
  _JAX_AVAILABLE = True
except Exception:
  chex = None  # type: ignore
  jax = None  # type: ignore
  jnp = None  # type: ignore
  _JAX_AVAILABLE = False

if not _JAX_AVAILABLE:
  import unittest as _unittest
  raise _unittest.SkipTest("JAX not available; skipping layers tests.")
import numpy as np
from videoprism import layers  # type: ignore


BaseJaxTestCase = absltest.TestCase if _JAX_AVAILABLE else unittest.TestCase


class LayersTest(BaseJaxTestCase, parameterized.TestCase):

  def test_identity(self):
    inputs = jnp.ones((4, 16, 8))
    outputs = layers.identity(inputs)
    self.assertEqual(inputs.shape, outputs.shape)
    self.assertTrue(jnp.array_equal(inputs, outputs))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(True, False)
  def test_layer_norm(self, direct_scale: bool):
    np_inputs = np.random.normal(1.0, 0.5, [10, 10, 10, 3]).astype(np.float32)
    inputs = jnp.asarray(np_inputs)
    prng_key = jax.random.PRNGKey(seed=123)
    ln = layers.LayerNorm(name='ln', direct_scale=direct_scale)

    @self.variant
    def var_fn():
      return ln.init_with_output(prng_key, inputs)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 2)
    self.assertEqual(inputs.shape, outputs.shape)

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  def test_feedforward_layer(self):
    np_inputs = np.random.normal(1.0, 0.5, [10, 10, 3]).astype(np.float32)
    inputs = jnp.asarray(np_inputs)
    prng_key = jax.random.PRNGKey(seed=123)
    ffn = layers.FeedForward(name='ffn', output_dim=20)

    @self.variant
    def var_fn():
      return ffn.init_with_output(prng_key, inputs)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 2)
    self.assertEqual(outputs.shape, (10, 10, 20))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(True, False)
  def test_transformer_feedforward(self, train: bool):
    batch_size, seq_len, input_dims = 4, 512, 8
    np_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]
    ).astype(np.float32)
    inputs = jnp.asarray(np_inputs)
    np_paddings = np.zeros([batch_size, seq_len], dtype=np.float32)
    input_paddings = jnp.asarray(np_paddings)
    prng_key = jax.random.PRNGKey(seed=123)
    ffwd = layers.TransformerFeedForward(
        name='ffwd', hidden_dim=32, activation_fn=layers.gelu
    )

    @self.variant
    def var_fn():
      return ffwd.init_with_output(
          prng_key, inputs, input_paddings, train=train
      )

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 6)
    self.assertEqual(outputs.shape, (batch_size, seq_len, input_dims))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(
      (16, 2, 5, False, [5, 16], [5, 2, 5]),
      (16, 2, 5, True, [5, 2, 5], [5, 16]),
      (256, 16, 16, True, [2, 16, 16], [2, 256]),
  )
  def test_mhd_projection(
      self,
      input_dim: int,
      num_heads: int,
      dim_per_head: int,
      is_output_projection: bool,
      inputs_shape: list[int],
      expected_outputs_shape: list[int],
  ):
    np_inputs = np.random.normal(1.5, 2.0, inputs_shape).astype(np.float32)
    inputs = jnp.asarray(np_inputs)
    prng_key = jax.random.PRNGKey(seed=123)
    mh = layers.AttentionProjection(
        name='mh',
        output_dim=input_dim,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        is_output_projection=is_output_projection,
    )

    @self.variant
    def var_fn():
      return mh.init_with_output(prng_key, inputs)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 2)
    self.assertEqual(outputs.shape, tuple(expected_outputs_shape))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  def test_per_dim_scale(self):
    np_inputs = np.random.normal(1.5, 2.0, [5, 4]).astype(np.float32)
    inputs = jnp.asarray(np_inputs)
    prng_key = jax.random.PRNGKey(seed=123)
    mdl = layers.PerDimScale(name='scale')

    @self.variant
    def var_fn():
      return mdl.init_with_output(prng_key, inputs)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 1)
    self.assertEqual(outputs.shape, (5, 4))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.product(
      enable_query_scale=[True, False],
      enable_per_dim_scale=[True, False],
      train=[True, False],
  )
  def test_mha(
      self,
      enable_query_scale: bool,
      enable_per_dim_scale: bool,
      train: bool,
  ):
    batch_size, seq_len, num_heads, mdl_dim = 3, 8, 4, 16
    query_vec = np.random.normal(size=[batch_size, seq_len, mdl_dim]).astype(
        np.float32
    )
    key_vec = np.random.normal(size=[batch_size, seq_len, mdl_dim]).astype(
        np.float32
    )
    value_vec = np.random.normal(size=[batch_size, seq_len, mdl_dim]).astype(
        np.float32
    )
    paddings = jnp.zeros(query_vec.shape[:-1], dtype=query_vec.dtype)
    atten_mask = layers.compute_attention_masks_for_fprop(query_vec, paddings)
    prng_key = jax.random.PRNGKey(seed=123)
    mha = layers.DotProductAttention(
        name='mha',
        hidden_dim=32,
        num_heads=num_heads,
        atten_logit_cap=20.0,
        internal_enable_query_scale=enable_query_scale,
        internal_enable_per_dim_scale=enable_per_dim_scale,
    )

    @self.variant
    def var_fn():
      return mha.init_with_output(
          prng_key, key_vec, query_vec, value_vec, atten_mask, train=train
      )

    (outputs, probs), params = var_fn()
    expected_num_weights = 8
    if enable_query_scale and enable_per_dim_scale:
      expected_num_weights += 1
    self.assertLen(jax.tree_util.tree_flatten(params)[0], expected_num_weights)
    self.assertEqual(outputs.shape, (batch_size, seq_len, mdl_dim))
    self.assertEqual(probs.shape, (batch_size, num_heads, seq_len, seq_len))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(True, False)
  def test_transformer_layer(self, train: bool):
    num_heads, batch_size, seq_len, dim = 8, 3, 12, 32
    np_inputs = np.random.normal(1.0, 0.5, [batch_size, seq_len, dim]).astype(
        'float32'
    )
    inputs = jnp.asarray(np_inputs)
    np_paddings = np.random.randint(0, 1, [batch_size, seq_len]).astype(
        'float32'
    )
    paddings = jnp.asarray(np_paddings)
    atten_mask = layers.compute_attention_masks_for_fprop(inputs, paddings)
    prng_key = jax.random.PRNGKey(seed=123)
    tfm = layers.Transformer(
        name='tfm',
        hidden_dim=128,
        num_heads=num_heads,
    )

    @self.variant
    def var_fn():
      return tfm.init_with_output(
          prng_key, inputs, paddings, atten_mask=atten_mask, train=train
      )

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 17)
    self.assertEqual(outputs.shape, (batch_size, seq_len, dim))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.product(
      scan=[True, False],
      train=[True, False],
  )
  def test_stacked_transformer_layer(self, scan: bool, train: bool):
    batch_size, seq_len, dim = 3, 12, 16
    np_inputs = np.random.normal(1.0, 0.5, [batch_size, seq_len, dim]).astype(
        'float32'
    )
    inputs = jnp.asarray(np_inputs)
    np_paddings = np.random.randint(0, 1, [batch_size, seq_len]).astype(
        'float32'
    )
    paddings = jnp.asarray(np_paddings)
    prng_key = jax.random.PRNGKey(seed=123)
    stacked_tfm = layers.StackedTransformer(
        name='stacked_tfm',
        hidden_dim=64,
        num_heads=8,
        num_layers=4,
        scan=scan,
    )

    @self.variant
    def var_fn():
      return stacked_tfm.init_with_output(
          prng_key, inputs, paddings, train=train
      )

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 17 if scan else 68)
    self.assertEqual(outputs.shape, (batch_size, seq_len, dim))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.product(
      num_queries=[1, 4],
      train=[True, False],
  )
  def test_atten_token_pooling_layer(
      self,
      num_queries: int,
      train: bool,
  ):
    batch_size, seq_len, num_heads, input_dim = 3, 8, 4, 16
    np_inputs = np.random.normal(
        1.5, 2.0, [batch_size, seq_len, input_dim]
    ).astype(np.float32)
    np_paddings = np.zeros([batch_size, seq_len], dtype=np.float32)
    inputs = jnp.asarray(np_inputs)
    input_paddings = jnp.asarray(np_paddings)
    prng_key = jax.random.PRNGKey(seed=123)
    pooler = layers.AttenTokenPoolingLayer(
        name='pooling',
        num_heads=num_heads,
        num_queries=num_queries,
    )

    @self.variant
    def var_fn():
      return pooler.init_with_output(
          prng_key, inputs, input_paddings, train=train
      )

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 12)
    self.assertEqual(outputs.shape, (batch_size, num_queries, input_dim))


if __name__ == '__main__':
  if _JAX_AVAILABLE:
    absltest.main()
  else:
    unittest.main()
