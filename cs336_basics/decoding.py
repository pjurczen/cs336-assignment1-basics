import argparse
import json
from typing import Optional

import torch

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.run_config import RunConfig
from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import TransformerLM


def decoding(args: argparse.Namespace):
    tokenizer = BpeTokenizer.from_vocab_merges_path(args.vocab_path, args.merges_path, [args.special_tokens])
    with open(args.model_config_path, "r") as f:
        model_config: RunConfig = RunConfig.model_validate(json.load(f))
    model: torch.nn.Module = TransformerLM(
        vocab_size=model_config.model_vocab_size,
        context_length=model_config.model_context_length,
        num_layers=model_config.model_num_layers,
        d_model=model_config.model_d_model,
        num_heads=model_config.model_num_heads,
        d_ff=model_config.model_d_ff,
        eps=model_config.model_eps,
        theta=model_config.model_rope_theta,
        device=torch.device(model_config.device),
        dtype=torch.float32
    )

    print("Loading model...")
    checkpoint: dict[str, any] = torch.load(args.model_checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    context: list[int] = []
    special_token_ids: set[int] = set(tokenizer.encode(args.special_tokens))

    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if len(prompt) == 0:
            continue
        encoded_prompt: list[int] = tokenizer.encode(prompt)
        context += encoded_prompt
        generated_tokens: int = 0

        while context[-1] not in special_token_ids and generated_tokens != args.max_tokens:
            context_cutoff: int = len(context) - model_config.model_context_length
            context_start: int = 0 if context_cutoff < 0 else context_cutoff
            context_window: list[int] = context[context_start:]

            model_input: torch.Tensor = torch.tensor(context_window, device=torch.device(model_config.device)).unsqueeze(0)
            with torch.no_grad():
                logits: torch.Tensor = model(model_input)  # (batch, sequence_length, vocab_size)

                next_token_logits = logits[0][-1]  # first (only) batch, last token

                next_token_id: int = sample_next_token(next_token_logits, args.temperature, args.top_p, args.top_k)
                context.append(next_token_id)
                generated_tokens += 1

                next_token: str = tokenizer.decode([next_token_id])
                print(next_token, end="", flush=True)
        print()


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: Optional[float] = None, top_k: Optional[int] = None) -> int:
    if temperature == 0:
        return torch.argmax(logits).item()
    scaled_prediction: torch.Tensor = logits / temperature
    probabilities = softmax(scaled_prediction, 0)
    values, indices = torch.sort(probabilities, descending=True)

    if top_k:
        values[top_k:] = 0

    if top_p:
        cumulative_probabilities = torch.cumsum(values, dim=0)
        difference = cumulative_probabilities - values
        mask = difference >= top_p
        values[mask] = 0

    index = torch.multinomial(values, num_samples=1)
    return indices[index].item()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--vocab_path", type=str, required=True)
    arg_parser.add_argument("--merges_path", type=str, required=True)
    arg_parser.add_argument("--special_tokens", type=str, required=True)
    arg_parser.add_argument("--model_config_path", type=str, required=True)
    arg_parser.add_argument("--model_checkpoint_path", type=str, required=True)
    arg_parser.add_argument("--temperature", type=float, default=1.0)
    arg_parser.add_argument("--top_p", type=float)
    arg_parser.add_argument("--top_k", type=int)
    arg_parser.add_argument("--max_tokens", type=int, default=1000)
    args = arg_parser.parse_args()

    decoding(args)
