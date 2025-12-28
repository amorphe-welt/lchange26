import torch
from abc import ABC, abstractmethod
from typing import Tuple, Literal, Optional
from transformers import AutoTokenizer, AutoModel
import logging


class ContextualSpanEmbedder(ABC):
    """
    Base class for extracting contextualized token embeddings
    from a token span in a sentence using HuggingFace models.

    This class supports:
    - HuggingFace model identifiers or local paths
    - Token-span-based token selection (using token indices)
    - Layer pooling (e.g., last, last4mean)
    - Subtoken pooling over all subtokens overlapping the span

    Conceptual model:
        sentence + token span → subtokens → pooled embedding

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local filesystem path.

    device : str, default="cuda"
        Torch device ("cuda", "cpu", "mps").

    Notes
    -----
    - Token spans are given as token indices: [start_token_idx, end_token_idx)
    - All subtokens whose token index overlaps the span are pooled
    - One instance corresponds to one embedding space
      (do NOT mix different models in the same AnnData)
    
    Typical usage
    -------------
    embedder = BertSpanEmbedder("bert-base-german-cased")
    vector = embedder.encode(
        sentence="Das ist ein Beispiel.",
        span=(2, 4)
    )
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
    ):
        self.device = device
        self.tokenizer = self._load_tokenizer(model_name_or_path)
        self.model = self._load_model(model_name_or_path).to(device).eval()

    # ------------------------------------------------------------------
    # Required subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str):
        pass

    @abstractmethod
    def _load_model(self, model_name_or_path: str):
        pass

    @abstractmethod
    def default_layer_pool(self) -> Literal["last", "last4mean"]:
        pass

    # ------------------------------------------------------------------
    # Pooling utilities
    # ------------------------------------------------------------------

    def _pool_layers(
        self,
        hidden_states: Tuple[torch.Tensor],
        layer_pool: Literal["last", "last4mean"],
    ) -> torch.Tensor:
        if layer_pool == "last":
            return hidden_states[-1]
        if layer_pool == "last4mean":
            return torch.stack(hidden_states[-4:]).mean(0)
        raise ValueError(f"Unknown layer_pool: {layer_pool}")

    def _pool_subtokens(
        self,
        token_embeddings: torch.Tensor,
        indices,
        method: Literal["mean", "max"] = "mean",
    ) -> torch.Tensor:
        vecs = token_embeddings[indices]
        if method == "mean":
            return vecs.mean(0)
        if method == "max":
            return vecs.max(0).values
        raise ValueError(f"Unknown subtoken_pool: {method}")

    def _span_to_subtokens(self, span, tokenized_sentence):
        """
        Map token indices to the subtoken indices that overlap with the span.
        Assumes span is provided as token indices.
        """
        start_token, end_token = span
        return list(range(start_token, end_token))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        sentence: str,
        span: Tuple[int, int],
        *,
        layer_pool: Optional[Literal["last", "last4mean"]] = None,
        subtoken_pool: Literal["mean", "max"] = "mean",
    ) -> torch.Tensor:
        """
        Compute a contextualized embedding for a token span.

        Parameters
        ----------
        sentence : str
            Input sentence.

        span : (int, int)
            Token index span [start_token_idx, end_token_idx) in Python slicing semantics.

        layer_pool : {"last", "last4mean"}, optional
            Layer pooling strategy.
            If None, uses model-specific default.

        subtoken_pool : {"mean", "max"}, default="mean"
            Pooling over subtokens overlapping the span.

        Returns
        -------
        torch.Tensor
            1D embedding vector (on CPU).

        Raises
        ------
        ValueError
            If no subtokens overlap the given span.
        """

        if layer_pool is None:
            layer_pool = self.default_layer_pool()

        # Tokenizing sentence
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
        )

        tokenized_sentence = encoding['input_ids'][0]  # Extract tokenized sentence
        offsets = encoding.pop("offset_mapping")[0]  # List of (start, end) character positions

        encoding = encoding.to(self.device)

        # Debugging: log tokenization results
        logging.debug(f"Tokenized sentence: {sentence}")
        logging.debug(f"Tokenized IDs: {tokenized_sentence}")
        logging.debug(f"Offsets: {offsets}")

        with torch.no_grad():
            outputs = self.model(**encoding, output_hidden_states=True)

        hidden = self._pool_layers(outputs.hidden_states, layer_pool)[0]

        # Find the token indices corresponding to the span
        indices = self._span_to_subtokens(span, tokenized_sentence)

        if not indices:
            logging.error(f"No subtokens overlap span {span} in sentence:\n{sentence}")
            raise ValueError(f"No subtokens overlap span {span} in sentence:\n{sentence}")

        # Pool the embeddings over the found subtokens
        return (
            self._pool_subtokens(hidden, indices, subtoken_pool)
            .detach()
            .cpu()
        )


# ----------------------------------------------------------------------
# Encoder-style models (BERT, RoBERTa, etc.)
# ----------------------------------------------------------------------

class BertSpanEmbedder(ContextualSpanEmbedder):
    """
    Span embedder for encoder-style models (BERT, RoBERTa, etc.).

    Defaults
    --------
    layer_pool = "last4mean"
    """

    def _load_tokenizer(self, model_name_or_path: str):
        return AutoTokenizer.from_pretrained(model_name_or_path)

    def _load_model(self, model_name_or_path: str):
        return AutoModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
        )

    def default_layer_pool(self):
        return "last4mean"


# ----------------------------------------------------------------------
# Decoder-style models (Qwen3, fine-tuned Qwen3, LLaMA-like)
# ----------------------------------------------------------------------

class DecoderSpanEmbedder(ContextualSpanEmbedder):
    """
    Span embedder for decoder-only models (Qwen3, LLaMA-like).

    Defaults
    --------
    layer_pool = "last"

    Notes
    -----
    - Uses trust_remote_code=True
    - Assumes standard HF hidden state outputs
    """

    def _load_tokenizer(self, model_name_or_path: str):
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    def _load_model(self, model_name_or_path: str):
        return AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            output_hidden_states=True,
        )

    def default_layer_pool(self):
        return "last"
