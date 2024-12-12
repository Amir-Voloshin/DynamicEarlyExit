# Dynamic Early Exit

This repository is built off of the repository provided for the implementation of [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710).

Files added and/or changed from original repository:
- self_speculation/llama_early_exit_utils.py
- self_speculation/llama_model_utils.py
- self_speculation/generator_basy.py
- self_speculation/autoregressive_generator.py

## Getting Started
- Clone repo:
```console
$ git clone git@github.com:facebookresearch/LayerSkip.git
$ cd LayerSkip
```

- Setup environment:
```console
$ conda create --name layer_skip python=3.10
$ conda activate layer_skip

$ pip install -r requirements.txt
```

- Access models:
In order to observe speedup, you need to access LLMs that have been trained using the LayerSkip recipe. We provide 6 checkpoints on [HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a) of different Llama models continually pretrained using the LayerSkip recipe:

    - [`facebook/layerskip-llama2-7B`](https://huggingface.co/facebook/layerskip-llama2-7B)
    - [`facebook/layerskip-llama2-13B`](https://huggingface.co/facebook/layerskip-llama2-13B)
    - [`facebook/layerskip-codellama-7B`](https://huggingface.co/facebook/layerskip-codellama-7B)
    - [`facebook/layerskip-codellama-34B`](https://huggingface.co/facebook/layerskip-codellama-34B)
    - [`facebook/layerskip-llama3-8B`](https://huggingface.co/facebook/layerskip-llama3-8B)
    - [`facebook/layerskip-llama3.2-1B`](https://huggingface.co/facebook/layerskip-llama3.2-1B)

In order to access each model:

1. Visit the model's corresponding link above, make sure you are logged on the HuggingFace website with your account.
2. Fill the request form and submit it. Approval may take a while and you should receive an email notification to notify you that permission to the model is granted.
3. Follow the steps [here](https://huggingface.co/docs/hub/en/security-tokens) to obtain a user access token.
4. In the command-line run `huggingface-cli login`, and you will be prompted to provide the token you have obtained in Step 3.

Once you run those steps, the commands below to run the LayerSkip checkpoints should work.

## Generate

To run a model in interactive mode using regular autoregressive decoding:
```console
$ torchrun generate.py --model facebook/layerskip-llama2-7B \
    --sample True \
    --max_steps 512
```

To perform dynamic early exit, you need to specify `--criteria`. Criteria options are: "cosine_similarity", "token_repeat", "entropy_based", "max_probability", or "convergence".

```console
$ torchrun generate.py --model facebook/layerskip-llama2-7B \
    --sample True \
    --max_steps 512 \
    --generation_strategy autoregressive \
    --exit_layer 16 \
    --criteria "cosine_similarity"
```

Tips:
- You may change `--model` to any HuggingFace model but in order to observe speedup with self-speculative decoding, use a model trained using the LayerSkip recipe, such as those we have [open sourced on HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a).
- By default we enable sampling. You may change the sampling behaviour using the `--sample`, `--temperature`, `--top_p`, and `--top_k` arguments.
- You may run `python generate.py --help` for details on different command-line arguments.

## Benchmark

To benchmark on a dataset:

```console
$ torchrun benchmark.py --model facebook/layerskip-llama2-7B \
    --dataset cnn_dm_summarization \
    --num_samples 100 \
    --generation_strategy autoregressive \
    --exit_layer 8 \
    --num_speculations 6 \
    --output_dir ./logs
```

Tips:
- You can specify different tasks by modifying the `--dataset` argument:
    - `cnn_dm_summarization`: CNN/DM Summarization
    - `xsum_summarization`: XSUM Summarization
    - `cnn_dm_lm`: CNN/DM Language Modeling (given the first few words of an article, generate the remaining article)
    - `human_eval`: HumanEval Coding
- By default, the tasks run as 0-shot. You can change to any specified `n`-shot by specifying the `--n_shot` argument.
- By default we enable sampling, while the results reported in the paper were greedy decoding without sampling. You may change the sampling behaviour using the `--sample`, `--temperature`, `--top_p`, and `--top_k` arguments.
- You may run `python benchmark.py --help` for details on different command-line arguments.

## Evaluate

We have integrated our generation scripts with [Eleuther Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) to enable a large number of tasks and properly post-process generated text.

```console
$ torchrun eval.py --model facebook/layerskip-llama2-7B \
    --tasks gsm8k \
    --limit 10 \
    --generation_strategy autoregressive \
    --exit_layer 8 \
    --num_speculations 6 \
    --output_dir ./logs
```

Tips:
- Note that with speculative decoding we can only obtain speedups from generation tasks (e.g., `gsm8k` or `cnn_dailymail`), while classificaton tasks, i.e., multiple choice question tasks (e.g., `piqa`, `social_iqa`) or True/False question tasks (e.g., `boolq`) will not lead to speedup.
- You can specify arbitrary number of tasks supported by Eleuther Evaluation Harness using the `--tasks` argument. To get a list of all of possible tasks, check this [link](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).
- Similar to the `generate.py` and `benchmark.py` scripts, you may specify different models, datasets, and sampling parameters
- You may run `python benchmark.py --help` for details on different command-line arguments.


## Using Docker

Kindly check [DOCKER.md](DOCKER.md) to setup the project using docker

