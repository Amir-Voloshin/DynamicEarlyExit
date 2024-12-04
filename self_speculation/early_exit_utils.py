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
    past_key_values,
    exit_query_cache,
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

    # Normalizing hidden states
    hidden_states = model.model.norm(hidden_states)

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
                past_key_values=past_key_values,
                exit_query_cache=exit_query_cache,
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
        # Normalizing hidden states
        hidden_states = model.model.norm(hidden_states)

        # Compute cosine similarity for each token
        cosine_sim = torch.sum(
            hidden_states * prev_hidden_states, dim=-1
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


def entropy_based_early_exit(
        hidden_states: torch.Tensor,
        prev_hidden_states: torch.Tensor,
        model,
        past_key_values,
        exit_query_cache,
        layer_idx: int,
        max_layers: int,
        initial_threshold: float = 2.0,
        final_threshold: float = 0.5,
        max_prob_threshold: Optional[float] = 0.7,  # Add a maximum probability threshold
) -> tuple:
    """
    Uses entropy to determine early exit, with a decreasing threshold as the network progresses.

    Args:
        hidden_states (torch.Tensor): Current hidden states from the model.
        model: Model instance to generate logits.
        past_key_values: Cache for past key values.
        exit_query_cache: Query cache for exits.
        layer_idx (int): Current layer index.
        max_layers (int): Total number of layers in the model.
        initial_threshold (float): Entropy threshold at the first layer.
        final_threshold (float): Entropy threshold at the last layer.

    Returns:
        tuple: ForwardResult and layer index if exiting early, or (None, None).
    """

    def dynamic_threshold(layer_idx, max_layers, initial, final, steepness=0.5):
        scale = (max_layers - layer_idx) / max_layers
        return initial * (scale ** steepness) + final * (1 - scale ** steepness)

    # Compute the dynamic threshold based on the layer index
    # Compute dynamic entropy threshold
    # threshold = (
    #         initial_threshold
    #         - (initial_threshold - final_threshold) * (layer_idx / max_layers)
    # )

    threshold = dynamic_threshold(layer_idx, max_layers, initial_threshold, final_threshold)

    # Calculate logits and softmax probabilities
    # if prev_hidden_states is not None:
    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)
    logits = torch.where(logits == 0, torch.randn_like(logits) * 1e-5, logits)
    last_token_logits = logits[:, -1, :]
    softmax_probs = torch.softmax(last_token_logits, dim=-1)

    # Clamp probabilities to avoid log(0) and nan
    softmax_probs = torch.clamp(softmax_probs, min=1e-9, max=1 - 1e-9)

    # Compute entropy
    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs), dim=-1).mean()

    # Log for debugging
    print(f"Layer {layer_idx}: Entropy = {entropy.item():.4f}, Threshold = {threshold:.4f}")

    print(f"Softmax probabilities (min: {softmax_probs.min().item()}, max: {softmax_probs.max().item()})")

    # Check for maximum probability threshold
    max_prob = softmax_probs.max().item()
    if max_prob_threshold is not None:
        if max_prob > max_prob_threshold:
            print(f"Early exit at Layer {layer_idx} with max probability = {max_prob:.4f}")
            return (
                ForwardResult(
                    logits=logits,
                    past_key_values=past_key_values,
                    exit_query_cache=exit_query_cache,
                ),
                layer_idx,
            )

    # Check for entropy threshold
    if entropy < threshold:
        print(f"Early exit at Layer {layer_idx} with entropy = {entropy:.4f}")
        return (
            ForwardResult(
                logits=logits,
                past_key_values=past_key_values,
                exit_query_cache=exit_query_cache,
            ),
            layer_idx,
        )

    return None, None


def entropy_gradient_early_exit(
        entropy_values: List[float],
        layer_idx: int,
        gradient_threshold: float = 0.02,
        window: int = 3,  # Window size for moving average
) -> bool:
    """
    Determines early exit based on entropy gradient stability.

    Args:
        entropy_values (List[float]): List of entropy values up to the current layer.
        layer_idx (int): Current layer index.
        gradient_threshold (float): Gradient threshold for triggering early exit.
        window (int): Number of layers to consider for stability.

    Returns:
        bool: True if the entropy gradient is stable, False otherwise.
    """
    # Skip gradient calculation for the first layer
    if layer_idx == 0:
        return False

    # Calculate entropy gradient
    entropy_gradient = abs(entropy_values[layer_idx] - entropy_values[layer_idx - 1])

    # Compute moving average of recent gradients (optional, for stability)
    if len(entropy_values) > window:
        recent_gradients = [
            abs(entropy_values[i] - entropy_values[i - 1])
            for i in range(layer_idx - window + 1, layer_idx + 1)
        ]
        avg_gradient = sum(recent_gradients) / len(recent_gradients)
    else:
        avg_gradient = entropy_gradient

    # Log for debugging
    print(f"Entropy Gradient = {entropy_gradient:.6f}, Moving Average = {avg_gradient:.6f} at Layer {layer_idx}")

    # Trigger early exit if gradient falls below the threshold
    return avg_gradient != 0.0 and avg_gradient < gradient_threshold