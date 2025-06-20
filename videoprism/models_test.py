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
import jax
from jax import numpy as jnp
import numpy as np
from videoprism import models


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(8, 16)
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

    outputs, _ = forward_fn(inputs)
    self.assertEqual(outputs.shape, (batch_size, num_frames * 16**2, 768))


if __name__ == '__main__':
  absltest.main()
