import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch.optim

from cs336_basics.adamw_optimizer import AdamWOptimizer
from cs336_basics.bpe_tokenizer import read_tokenizer_config
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient import gradient_clipping
from cs336_basics.learning_rate import cosine_lr_schedule
from cs336_basics.transformer_lm import TransformerLM


def training(args: argparse.Namespace):
    iteration: int = 1

    run_dir = os.path.join(args.runs_dir, args.run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run directory: {run_dir}")

    model: torch.nn.Module = TransformerLM(
        vocab_size=args.model_vocab_size,
        context_length=args.model_context_length,
        num_layers=args.model_num_layers,
        d_model=args.model_d_model,
        num_heads=args.model_num_heads,
        d_ff=args.model_d_ff,
        eps=args.model_eps,
        theta=args.model_rope_theta,
        device=args.device,
        dtype=torch.float32
    )
    model = torch.compile(model, backend="aot_eager")

    optimizer: torch.optim.Optimizer = AdamWOptimizer(
        params=model.parameters(),
        lr=args.optimizer_lr_min,
        weight_decay=args.optimizer_weight_decay,
        betas=(args.optimizer_beta1, args.optimizer_beta2),
        eps=args.optimizer_eps,
    )

    elapsed_time = None
    if args.checkpoint_src:
        print("Loading checkpoint...")
        iteration, elapsed_time = load_checkpoint(args.checkpoint_src, model, optimizer)

        print(f"Resuming from iteration {iteration}/{args.iter}")
        iteration += 1

    train_dataset = np.load(file=args.train_data, mmap_mode='r')
    val_dataset = np.load(file=args.val_data, mmap_mode='r') if args.val_data else None

    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as metric_file:

        t0 = time.perf_counter()
        if elapsed_time is not None:
            t0 -= elapsed_time

        for it in range(iteration, args.iter + 1):
            inputs, targets = get_batch(dataset=train_dataset, batch_size=args.batch_size, context_length=args.model_context_length, device=args.device,
                                        dtype=torch.long)

            prediction: torch.Tensor = model(inputs)

            loss: torch.Tensor = cross_entropy(prediction, targets)
            loss.backward()

            l2_norm: float = gradient_clipping(parameters=model.parameters(), max_l2_norm=args.gradient_max_l2_norm, eps=args.gradient_eps)

            if it % args.log_interval == 0 or it == args.iter:
                print(f"{it}/{args.iter} Training loss: {loss.item()}")
            if it % (10 * args.log_interval) == 0 or it == args.iter:
                print(f"{it}/{args.iter} L2-norm: {l2_norm}")

            lr: float = cosine_lr_schedule(
                t=it,
                lr_max=args.optimizer_lr_max,
                lr_min=args.optimizer_lr_min,
                warmup_iters=args.optimizer_warmup_iters,
                cosine_cycle_iters=args.optimizer_cosine_cycle_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            if it % args.log_interval == 0 or it == args.iter:
                synchronize(args.device)
                t1 = time.perf_counter()

                print(f"Elapsed time: {(t1 - t0):.0f} s")

                metrics: dict = {
                    "step": it, "time": round(t1 - t0, 2), "train_loss": loss.item(), "lr": lr
                }
                metric_file.write(json.dumps(metrics) + "\n")
                metric_file.flush()

            with torch.no_grad():
                if val_dataset is not None and (it % args.val_interval == 0 or it == args.iter):
                    val_loss_sum: float = 0
                    for i in range(args.val_batches):
                        val_inputs, val_targets = get_batch(dataset=val_dataset, batch_size=args.batch_size, context_length=args.model_context_length,
                                                            device=args.device, dtype=torch.long)

                        val_prediction = model(val_inputs)
                        val_loss_sum += cross_entropy(val_prediction, val_targets).item()

                    val_loss = val_loss_sum / args.val_batches

                    t2 = time.perf_counter()

                    metrics: dict = {
                        "step": it, "time": t2 - t0, "val_loss": val_loss,
                    }
                    metric_file.write(json.dumps(metrics) + "\n")
                    metric_file.flush()

                    print(f"{it}/{args.iter} Validation loss: {val_loss}")

            if it % args.checkpoints_iter == 0 or it == args.iter:
                t3 = time.perf_counter()
                save_checkpoint(model, optimizer, it, os.path.join(checkpoints_dir, f"ckpt_{it}.pt"), t3 - t0)
                print(f"{it}/{args.iter} Saving checkpoint...")


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--optimizer_lr_min", type=float, default=1e-4)
    arg_parser.add_argument("--optimizer_lr_max", type=float, default=1e-3)
    arg_parser.add_argument("--optimizer_warmup_iters", type=int, default=None)
    arg_parser.add_argument("--optimizer_cosine_cycle_iters", type=int, default=None)
    arg_parser.add_argument("--optimizer_weight_decay", type=float, default=0.01)
    arg_parser.add_argument("--optimizer_beta1", type=float, default=0.9)
    arg_parser.add_argument("--optimizer_beta2", type=float, default=0.999)
    arg_parser.add_argument("--optimizer_eps", type=float, default=1e-8)
    arg_parser.add_argument("--model_context_length", type=int, default=1024)
    arg_parser.add_argument("--model_num_layers", type=int, default=48)
    arg_parser.add_argument("--model_d_model", type=int, default=1600)
    arg_parser.add_argument("--model_d_ff", type=int, default=4288)
    arg_parser.add_argument("--model_num_heads", type=int, default=25)
    arg_parser.add_argument("--model_eps", type=float, default=1e-5)
    arg_parser.add_argument("--model_rope_theta", type=int, default=10000)
    arg_parser.add_argument("--gradient_max_l2_norm", type=float, default=1.0)
    arg_parser.add_argument("--gradient_eps", type=float, default=1e-6)
    arg_parser.add_argument("--iter", type=int, default=100)
    arg_parser.add_argument("--device", type=str, default="mps")
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--tokenizer_dir", type=str, required=True)
    arg_parser.add_argument("--train_data", type=str, required=True)
    arg_parser.add_argument("--val_data", type=str)
    arg_parser.add_argument("--val_batches", type=int, default=32)
    arg_parser.add_argument("--val_interval", type=int, default=200)
    arg_parser.add_argument("--log_interval", type=int, default=100)
    arg_parser.add_argument("--runs_dir", type=str, default="runs")
    arg_parser.add_argument("--run_name", type=str, default=None)
    arg_parser.add_argument("--checkpoints_iter", type=int, default=100)
    arg_parser.add_argument("--checkpoint_src", type=str, default=None)
    args = arg_parser.parse_args()

    if args.optimizer_warmup_iters is None:
        args.optimizer_warmup_iters = args.iter // 50
    if args.optimizer_cosine_cycle_iters is None:
        args.optimizer_cosine_cycle_iters = args.iter
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.model_vocab_size = read_tokenizer_config(args.tokenizer_dir)["vocab_size"]

    assert args.optimizer_lr_min <= args.optimizer_lr_max
    assert args.optimizer_warmup_iters < args.optimizer_cosine_cycle_iters <= args.iter

    training(args)
