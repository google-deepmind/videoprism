# Copyright 2026 VideoPrism Authors.
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

"""Tests for utility functions."""

from absl.testing import absltest
from videoprism import utils


class UtilsTest(absltest.TestCase):

  def test_canonicalize_text(self):
    self.assertEqual(utils.canonicalize_text("Hello, World!"), "hello world.")
    self.assertEqual(utils.canonicalize_text("Hello,World.."), "hello world.")
    self.assertEqual(utils.canonicalize_text("  Hello   WORLD"), "hello world.")

  def test_traverse_with_names_treats_strings_as_leaves(self):
    tree = {"caption": "hello world"}
    self.assertEqual(
        list(utils.traverse_with_names(tree)),
        [("caption", "hello world")],
    )

  def test_traverse_with_names_treats_bytes_as_leaves(self):
    tree = {"blob": b"abc"}
    self.assertEqual(list(utils.traverse_with_names(tree)), [("blob", b"abc")])


if __name__ == "__main__":
  absltest.main()
