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

        # llama model normalization of hidden states
        hidden_states = model.model.norm(hidden_states)

        # l2 normalization for cosine similarity
        hidden_states_l2 = hidden_states / hidden_states.norm(
            dim=-1, keepdim=True
        )  # Shape: (batch_size, seq_length, hidden_dim)
        prev_hidden_states_l2 = prev_hidden_states / prev_hidden_states.norm(
            dim=-1, keepdim=True
        )  # Same shape as above

        # Compute cosine similarity for each token
        cosine_sim = torch.sum(
            hidden_states_l2 * prev_hidden_states_l2, dim=-1
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

def convergence_early_exit(
    hidden_states: torch.Tensor,
    prev_hidden_states: torch.Tensor,
    model,
    past_key_values,
    exit_query_cache,
    layer_idx: int,
    delta_threshold: float = 0.01,
) -> tuple:
    """
    Creates a convergence early exit strategy. When the increments in the logits confidence
    are not bigger than 1%, it stops the generation.
    """
    # Skip if no previous hidden states
    if prev_hidden_states is None:
        return None, None

    # Normalizing hidden states
    hidden_states = model.model.norm(hidden_states)

    # Get the predicted token from the current layer's hidden states
    logits = model.lm_head(hidden_states)
    last_token_logits = logits[:, -1, :]
    softmax_probs = F.softmax(last_token_logits, dim=-1)
    predicted_token_prob = softmax_probs.max().item()

    # Print the softmax probability of the predicted token
    # print(f"Layer {layer_idx}: Predicted token softmax probability: {predicted_token_prob:.4f}")

    # Store probabilities per layer
    if not hasattr(convergence_early_exit, 'layer_probs'):
        convergence_early_exit.layer_probs = {}
    
    convergence_early_exit.layer_probs[layer_idx] = predicted_token_prob

    # Check convergence only after a few layers
    if layer_idx >= 2:
        prev_layer_prob = convergence_early_exit.layer_probs.get(layer_idx - 1, 0)
        # print(f"Layer {layer_idx}: Predicted prev token softmax probability: {prev_layer_prob:.4f}")
        confidence_increment = abs(predicted_token_prob - prev_layer_prob)
        
        # print(f"{confidence_increment=}")
        if abs(confidence_increment) <= delta_threshold:
            # print(f"Early exit at layer {layer_idx} with confidence increment {confidence_increment:.4f}")
            return (
                ForwardResult(
                    logits=logits,
                    past_key_values=past_key_values,
                    exit_query_cache=exit_query_cache,
                ),
                layer_idx,
            )
    return None, None

def max_prob_early_exit(
    hidden_states: torch.Tensor,
    model,
    past_key_values,
    exit_query_cache,
    layer_idx: int,
    max_layers: int,
    initial_threshold: float = 0.99,
    final_threshold: float = 0.75,
    scale: float = 1.0,
) -> tuple:
    """
    Early exit based on dynamic max probability threshold.

    Args:
        hidden_states (torch.Tensor): Model's hidden states.
        model: Model instance.
        past_key_values: Cache of past key values.
        exit_query_cache: Query cache for exits.
        layer_idx (int): Current layer index.
        max_layers (int): Total number of layers.
        initial_threshold (float): Starting threshold for max probability.
        final_threshold (float): Final threshold for max probability.
        scale (float): Controls the curve of the dynamic threshold.

    Returns:
        tuple: ForwardResult and layer index if exiting early, else (None, None).
    """

    def dynamic_max_prob_threshold(layer_idx, max_layers, initial, final, scale):
        layer_ratio = (max_layers - layer_idx) / max_layers
        return initial * (layer_ratio**scale) + final * (1 - layer_ratio**scale)

    max_prob_threshold = dynamic_max_prob_threshold(
        layer_idx, max_layers, initial_threshold, final_threshold, scale
    )

    # Normalize hidden states and compute logits
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    # Replace zero logits to avoid log(0) issues
    logits = torch.where(logits == 0, torch.randn_like(logits) * 1e-5, logits)
    last_token_logits = logits[:, -1, :]
    softmax_probs = torch.softmax(last_token_logits, dim=-1)

    # Clamp probabilities to ensure no zero values
    softmax_probs = torch.clamp(softmax_probs, min=1e-12, max=1.0 - 1e-12)

    # Compute log probabilities
    log_probs = torch.log(softmax_probs)

    # Find the token with the maximum probability
    max_prob, max_prob_token = torch.max(softmax_probs, dim=-1)

    # Check for maximum probability threshold
    if max_prob_threshold is not None:
        if max_prob.item() > max_prob_threshold:
            # print(f"Early exit at Layer {layer_idx} with max probability = {max_prob.item():.4f}")
            # print(f"Token triggering early exit: {max_prob_token.item()} with probability = {max_prob.item():.4f}")
            return (
                ForwardResult(
                    logits=logits,
                    past_key_values=past_key_values,
                    exit_query_cache=exit_query_cache,
                ),
                layer_idx,
                max_prob_token.item(),  # Return the token index
            )

    return None, None, None


def entropy_based_early_exit(
    hidden_states: torch.Tensor,
    model,
    past_key_values,
    exit_query_cache,
    layer_idx: int,
    max_layers: int,
    initial_threshold: float = 11.0,
    final_threshold: float = 10.5,
    temperature: float = 3.0,
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
        temperature (float): Entropy temperature.

    Returns:
        tuple: ForwardResult and layer index if exiting early, or (None, None).
    """

    def dynamic_threshold(layer_idx, max_layers, initial, final, steepness=0.5):
        scale = (max_layers - layer_idx) / max_layers
        return initial * (scale**steepness) + final * (1 - scale**steepness)

    threshold = dynamic_threshold(
        layer_idx, max_layers, initial_threshold, final_threshold
    )

    # Normalize hidden states and compute logits
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    # Stabilized entropy calculation
    stabilized_logits = (logits - logits.max(dim=-1, keepdim=True).values) / temperature
    softmax_probs = torch.softmax(
        stabilized_logits, dim=-1
    ).float()  # Cast back to float32
    softmax_probs = torch.clamp(
        softmax_probs, min=1e-12
    )  # Ensure no probabilities are too small
    log_probs = torch.log(softmax_probs)
    entropy = -torch.sum(softmax_probs * log_probs, dim=-1).mean()

    # Check for entropy threshold
    if entropy < threshold:
        return (
            ForwardResult(
                logits=logits,
                past_key_values=past_key_values,
                exit_query_cache=exit_query_cache,
            ),
            layer_idx,
        )

    return None, None
