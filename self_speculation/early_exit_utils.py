import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ForwardResult:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    exit_query_cache: Optional[List[torch.Tensor]] = None


def compute_cosine_similarity(
    hidden_states: torch.Tensor,
    prev_hidden_states: torch.Tensor,
    similarity_threshold: float,
    model,
    past_key_values,
    exit_query_cache,
    layer_idx: int,
) -> tuple:
    """
    Checks cosine similarity between current and previous hidden states.
    Exits early if the similarity exceeds the threshold.

    Args:
        hidden_states (torch.Tensor): Current hidden states from the model.
        prev_hidden_states (torch.Tensor): Hidden states from the previous layer.
        similarity_threshold (float): Threshold for triggering early exit.
        model (transformers.LlamaForCausalLM): Model instance to generate logits.
        past_key_values (Optional): Cache for past key values.
        exit_query_cache (Optional): Query cache for exits.
        layer_idx (int): Current layer index.

    Returns:
        tuple: ForwardResult and layer index if exiting early, or (None, None).
    """
    if prev_hidden_states is not None:
        # Normalize hidden states for cosine similarity calculation
        norm_hidden_states = F.normalize(hidden_states, dim=-1)
        norm_prev_hidden_states = F.normalize(prev_hidden_states, dim=-1)

        # Compute cosine similarity for each token
        cosine_sim = torch.sum(
            norm_hidden_states * norm_prev_hidden_states, dim=-1
        )  # Shape: (batch_size, seq_length)

        # Exit early if similarity exceeds the threshold for all tokens in a batch
        if torch.all(cosine_sim > similarity_threshold):
            logits = model.lm_head(hidden_states)

            return (
                ForwardResult(
                    logits=logits,
                    past_key_values=past_key_values,
                    exit_query_cache=exit_query_cache,
                ),
                layer_idx,
            )

    return None, None  # Continue processing
