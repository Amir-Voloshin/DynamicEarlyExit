import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ForwardResult:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    exit_query_cache: Optional[List[torch.Tensor]] = None


def token_repeat_early_exit(
    model,
    hidden_states,
    prev_token: Optional[torch.Tensor],
    token_repeats: int,
    layer_idx: int,
    repeats: int,
) -> Tuple[Optional[ForwardResult], int, torch.Tensor, int]:
    """
    Checks for token repetition across layers and triggers an early exit if the
    predicted token is repeated for 3 consecutive layers.

    Args:
        model: The model instance.
        hidden_states: The hidden states from the current layer.
        prev_token: The predicted token from the previous layer.
        token_repeats: The count of repeated tokens across consecutive layers.
        layer_idx: The current layer index.
        repeats: Number of times a token needs to be repeated for early exit

    Returns:
        - ForwardResult if early exit is triggered, otherwise None.
        - Current layer index (layer_idx).
        - Updated `prev_token` for the next layer.
        - Updated `token_repeats` count.
    """
    # Get the predicted token from the current layer's hidden states (last token predicted).
    logits = model.lm_head(hidden_states)
    predicted_token = logits.argmax(dim=-1)[
        :, -1
    ]  # Take the token with max probability.

    # Check if the predicted token is the same as the previous layer's prediction.
    if prev_token is not None and (predicted_token == prev_token).all():
        token_repeats += 1
    else:
        token_repeats = 0

    # Exit early if the same token is predicted for 3 consecutive layers.
    if token_repeats >= repeats:
        # Early exit triggered due to repeated token.
        return (
            ForwardResult(
                logits=logits,
                past_key_values=model.past_key_values,
                exit_query_cache=getattr(model, "exit_query_cache", None),
            ),
            layer_idx,
            prev_token,
            token_repeats,
        )

    # Update prev_token for the next layer.
    prev_token = predicted_token

    return None, layer_idx, prev_token, token_repeats


def cosine_similarity_early_exit(
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
