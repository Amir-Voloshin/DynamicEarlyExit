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

        exit_query_cache = None
        for _ in range(generation_config.max_steps):
            if generation_config.exit_layer > 0 or generation_config.criteria:
                model_output, exit_layer = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                    early_exit_criteria=generation_config.criteria,  
                    delta_threshold=generation_config.delta_threshold,
                    early_exit_criteria=generation_config.criteria,
                    repeats=generation_config.repeats,
                    similarity_threshold=generation_config.conf,
                )

                # saving exit layer for token
                exited_layers.append(exit_layer)

            else:
                model_output = forward(
                    model,
                    input_ids,
                    past_key_values,
                )
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
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
            output_ids.append(next_token)
            # Don't concatenate `next_token` to original `input_ids` since we're using
            # the KV cache (`past_key_values`) to speed up generation.
            input_ids = torch.tensor([[next_token]]).to(input_ids)

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
                    writer.writerow([token_number, layer, generation_config.criteria])

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )
