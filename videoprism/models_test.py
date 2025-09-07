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

"""Tests for VideoPrism models."""

from absl.testing import absltest
from absl.testing import parameterized
import unittest
try:
  import jax  # type: ignore
  from jax import numpy as jnp  # type: ignore
  _JAX_AVAILABLE = True
except Exception:
  jax = None  # type: ignore
  jnp = None  # type: ignore
  _JAX_AVAILABLE = False

if not _JAX_AVAILABLE:
  import unittest as _unittest
  raise _unittest.SkipTest("JAX not available; skipping models tests.")
import numpy as np
from videoprism import models
from videoprism import tokenizers


BaseJaxTestCase = absltest.TestCase if _JAX_AVAILABLE else unittest.TestCase


class ModelsTest(BaseJaxTestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('videoprism_public_v1_base', True),
      ('videoprism_public_v1_large', True),
      ('videoprism_public_v1_giant', False),
  )
  def test_has_model(self, model_name, exists):
    self.assertEqual(models.has_model(model_name), exists)

  @parameterized.parameters(8, 16)
  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  def test_videoprism(self, num_frames):
    batch_size = 1
    np_inputs = np.random.normal(
        0.0, 0.1, [batch_size, num_frames, 288, 288, 3]
    ).astype('float32')
    inputs = jnp.asarray(np_inputs)
    prng_key = jax.random.PRNGKey(seed=123)

    mdl = models.videoprism_v1_base()
    mdl_params = mdl.init(prng_key, inputs, train=False)

    @jax.jit
    def forward_fn(mdl_inputs):
      return mdl.apply(mdl_params, mdl_inputs, train=False)

    embeddings, _ = forward_fn(inputs)
    self.assertEqual(embeddings.shape, (batch_size, num_frames * 16**2, 768))

  @unittest.skipIf(not _JAX_AVAILABLE, "JAX not available")
  def test_videoprism_lvt(self):
    batch_size, num_frames = 1, 16
    np_inputs = np.random.normal(
        0.0, 0.1, [batch_size, num_frames, 288, 288, 3]
    ).astype('float32')
    inputs = jnp.asarray(np_inputs)
    np_text_token_ids = np.random.randint(
        0, 32_000, [batch_size, models.TEXT_MAX_LEN]
    ).astype('int32')
    text_token_ids = jnp.asarray(np_text_token_ids)
    np_text_paddings = np.zeros(
        [batch_size, models.TEXT_MAX_LEN], dtype='float32'
    )
    np_text_paddings[:, models.TEXT_MAX_LEN // 2 :] = 1
    text_paddings = jnp.asarray(np_text_paddings)
    prng_key = jax.random.PRNGKey(seed=123)

    mdl = models.videoprism_lvt_v1_base()
    mdl_params = mdl.init(
        prng_key, inputs, text_token_ids, text_paddings, train=False
    )

    @jax.jit
    def forward_fn(mdl_inputs, mdl_text_token_ids, mdl_text_paddings):
      return mdl.apply(
          mdl_params,
          mdl_inputs,
          mdl_text_token_ids,
          mdl_text_paddings,
          train=False,
      )

    vision_embeddings, text_embeddings, _ = forward_fn(
        inputs, text_token_ids, text_paddings
    )
    self.assertEqual(vision_embeddings.shape, (batch_size, 768))
    self.assertEqual(text_embeddings.shape, (batch_size, 768))

  def test_tokenize_texts(self):
    import os
    spm_path = os.path.join(
        os.path.dirname(__file__), 'assets', 'testdata', 'test_spm.model'
    )
    model = tokenizers.SentencePieceTokenizer(spm_path)
    ids, paddings = models.tokenize_texts(
        model,
        ['blah', 'blah blah', 'blah blah blah'],
        max_length=6,
        add_bos=False,
        canonicalize=False,
    )
    np.testing.assert_array_equal(
        ids,
        [
            [80, 180, 60, 0, 0, 0],
            [80, 180, 60, 80, 180, 60],
            [80, 180, 60, 80, 180, 60],
        ],
    )
    np.testing.assert_array_equal(
        paddings, [[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    )


if __name__ == '__main__':
  if _JAX_AVAILABLE:
    absltest.main()
  else:
    unittest.main()
