# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional

import torch
import csv

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import (
    decode_next_token,
    forward,
    forward_early,
)


class AutoRegressiveGenerationStrategy(GenerationStrategy):
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
        csv_file_path: str = "tokens_by_layer.csv",  # CSV file path
        logits_processors: Optional[
            transformers.generation.logits_process.LogitsProcessorList
        ] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:

        """Variant of `generate` with inputs/outputs formatted as token_ids."""
        past_key_values = None

        input_ids: torch.Tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        # recording which layers each token exited at
        exited_layers = []

        # Collect predictions across tokens
        all_predictions = []

        exit_query_cache = None
        for token_number in range(generation_config.max_steps):
            if generation_config.exit_layer > 0 or generation_config.criteria:
                model_output, exit_layer = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                    early_exit_criteria=generation_config.criteria,
                    repeats=generation_config.repeats,
                    similarity_threshold=generation_config.conf,
                )

                # saving exit layer for token
                exited_layers.append(exit_layer)

            else:
                model_output, predictions = forward(
                    model=model, input_ids=input_ids, past_key_values=past_key_values
                )  # Collect predictions for this token
                all_predictions.append(predictions)  # Store predictions

            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
            past_key_values = model_output.past_key_values
            next_token, _ = decode_next_token(
                logits=logits,
                token_idx=-1,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if next_token == eos_token_id:
                break
            if stopping_criteria:
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
            output_ids.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(input_ids)

        # Write all predictions to CSV after generation completes
        # if torch.distributed.get_rank() == 0:  # Ensure only rank 0 writes
        with open(csv_file_path, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            # Prepare header
            max_tokens = max(len(predictions) for predictions in all_predictions)
            header = ["Layer"] + [f"Token {i+1}" for i in range(max_tokens)]
            csv_writer.writerow(header)

            # Write layer rows
            for layer_idx in range(len(model.model.layers)):
                row = [f"Layer {layer_idx + 1}"]
                for predictions in all_predictions:
                    if layer_idx < len(predictions):
                        row.append(
                            predictions[layer_idx][1]
                        )  # Use token from predictions
                    else:
                        row.append("")
                csv_writer.writerow(row)

            # if performing dynamic early exit, save results to a csv
            if generation_config.criteria:
                # Specify the CSV filename
                csv_filename = "exited_layers.csv"

                # Write results to CSV
                with open(csv_filename, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    # Include token number in the header
                    writer.writerow(["Token Number", "Exit Layer", "Criteria"])
                    # Iterate over exited_layers with their corresponding indices
                    for token_number, layer in enumerate(
                        exited_layers, start=1
                    ):  # Token numbers start from 1
                        writer.writerow(
                            [token_number, layer, generation_config.criteria]
                        )

            return GenerationStrategyResult(
                predicted_tokens=output_ids,
                acceptance_rate=None,
            )
