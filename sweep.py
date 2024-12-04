# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List, Tuple
import pandas as pd
import transformers
from datetime import datetime
import os
import tabulate
import torch

from arguments import Arguments, simple_parse_args_string
from benchmark import benchmark, load_model_and_tokenizer, process_cli_arguments, setup, BenchmarkArguments
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)


def sweep(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    results: List[Dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    # TODO: make start, end, step arguments to the script
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    for exit_layer in range(1, len(model.model.layers) // 2, 1):
        for num_speculations in range(1, 13, 1):
            generation_config.exit_layer = exit_layer
            generation_config.num_speculations = num_speculations

            metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, args.seed)

            results.append({
                "exit_layer": exit_layer,
                "num_speculations": num_speculations,
                "acceptance_rate": metric_result['acceptance_rate']['mean'],
                "total_time": metric_result['total_time']['mean'],
                "time_per_token": metric_result['time_per_token']['mean'],
                "tokens_per_second": metric_result['tokens_per_second']['mean'],
            })
            df = pd.DataFrame(results) 
            # Update table every iteration
            df.to_csv(output_fname, index=False)
            print(f"exit_layer: {exit_layer}, num_speculations: {num_speculations}, time_per_token: {metric_result['time_per_token']['mean']}")

    # Print summary table
    print("\n")
    header = results[0].keys()
    rows =  [x.values() for x in results]
    print(tabulate.tabulate(rows, header))


def sweep_chat(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    # TODO: make start, end, step arguments to the script
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    else:
        raise Exception(
            f"Unsupported generation strategy: {generation_config.generation_strategy}"
        )

    exit_layer = 16
    num_speculations = 1

    generation_config.exit_layer = exit_layer
    generation_config.num_speculations = num_speculations

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    print("Enter prompts to generate responses (type 'exit' to quit):")
    while True:
        user_input = input("\n[Prompt]: ")
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        # Generate response
        response = generator.generate(prompt=user_input, generation_config=generation_config)
        print(f"[Prediction]: {response.decoded_prediction}")


if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    sweep(args, benchmark_arguments, generation_config, f"{args.output_dir}/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    # sweep_chat(args, benchmark_arguments, generation_config, f"{args.output_dir}/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")