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

"""Tokenizers for text encoders.

This module avoids a hard dependency on TensorFlow. TensorFlow-backed ops are
available when TensorFlow is installed; otherwise, pure-Python tokenization
works, and TensorFlow-specific helpers will raise a clear error.
"""

from collections.abc import Sequence
from typing import Protocol

import os
from typing import Any

try:  # Optional TensorFlow dependency
  import tensorflow as tf  # type: ignore
  from tensorflow.io import gfile  # type: ignore
  _TF_AVAILABLE = True
except Exception:  # pragma: no cover - optional path
  tf = None  # type: ignore[assignment]
  gfile = None  # type: ignore[assignment]
  _TF_AVAILABLE = False

import sentencepiece

SentencePieceProcessor = sentencepiece.SentencePieceProcessor


class Tokenizer(Protocol):
  """Tokenizer interface."""

  def to_int(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> list[int] | list[list[int]]:
    """Tokenizes `text` into a list of integer tokens.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A list or list-of-list of tokens.
    """

  def to_int_tf_op(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> Any:
    """Same as `to_int()`, but as TF ops to be used in data pipelines.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A tf.Tensor of tokens.
    """

  @property
  def pad_token(self) -> int:
    """Token id of padding token."""

  @property
  def eos_token(self) -> int:
    """Token id of end-of-sentence token."""

  @property
  def bos_token(self) -> int:
    """Token id of beginning-of-sentence token."""

  @property
  def vocab_size(self) -> int:
    """Returns the size of the vocabulary."""


class SentencePieceTokenizer(Tokenizer):
  """Wraps a SentencePiece model for tokenization."""

  def __init__(self, model_path):
    """Initializes the tokenizer.

    Args:
      model_path: A path to load the SentencePiece model.
    """
    # Support both local files and GCS paths when TensorFlow is available.
    if isinstance(model_path, os.PathLike):
      model_path = os.fspath(model_path)

    if isinstance(model_path, str) and model_path.startswith("gs://"):
      if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to read GCS paths (gs://...). "
            "Install tensorflow or provide a local file path."
        )
      with gfile.GFile(model_path, "rb") as f:  # type: ignore[arg-type]
        model_bytes = f.read()
    else:
      with open(model_path, "rb") as f:
        model_bytes = f.read()

    self._model = SentencePieceProcessor()
    self._model.LoadFromSerializedProto(model_bytes)

  def to_int(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> list[int] | list[list[int]]:
    """Tokenizes `text` into a list of integer tokens.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A list or list-of-list of tokens.
    """

    def _single(s: str) -> list[int]:
      return (
          ([self.bos_token] if bos else [])
          + self._model.EncodeAsIds(s)
          + ([self.eos_token] if eos else [])
      )

    if isinstance(text, str):
      return _single(text)
    return list([_single(s) for s in text])

  def to_int_tf_op(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> Any:
    """Same as `to_int()`, but as TF ops to be used in data pipelines.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A tf.Tensor or tf.RaggedTensor of tokens.
    """
    if not _TF_AVAILABLE:
      raise ImportError(
          "TensorFlow is not available. `to_int_tf_op` requires TensorFlow."
      )

    text = tf.convert_to_tensor(text)  # type: ignore[call-arg]
    if text.ndim == 0:

      def fn(txt):
        """Tokenizes a single string."""
        s = txt.numpy().decode()
        return tf.constant(self.to_int(s, bos=bos, eos=eos), tf.int32)

      return tf.py_function(fn, [text], tf.int32)  # type: ignore[arg-type]
    else:

      def fn(txt):
        """Tokenizes a list of strings."""
        strings = [s.decode() for s in txt.numpy().tolist()]
        toks = self.to_int(strings, bos=bos, eos=eos)
        return tf.ragged.constant(toks)

      out_type = tf.RaggedTensorSpec([text.shape[0], None], tf.int32)
      return tf.py_function(fn, [text], Tout=out_type)  # type: ignore[arg-type]

  @property
  def pad_token(self) -> int:
    """Token id of padding token."""
    return self._model.pad_id()

  @property
  def eos_token(self) -> int:
    """Token id of end-of-sentence token."""
    return self._model.eos_id()

  @property
  def bos_token(self) -> int:
    """Token id of beginning-of-sentence token."""
    return self._model.bos_id()

  @property
  def vocab_size(self) -> int:
    """Returns the size of the vocabulary."""
    return self._model.GetPieceSize()
