import argparse

ITER_CHECKPOINT = 200


def training(batch_size: int, lr: float):
    pass


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
    arg_parser.add_argument("--model_vocab_size", type=int, default=50257)
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
    arg_parser.add_argument("--train_data", type=str, required=True)
    arg_parser.add_argument("--val_data", type=str)
    arg_parser.add_argument("--val_batches", type=int, default=32)
    arg_parser.add_argument("--val_interval", type=int, default=200)
    arg_parser.add_argument("--log_interval", type=int, default=100)
    arg_parser.add_argument("--checkpoints_out", type=str, required=True)
    arg_parser.add_argument("--checkpoints_iter", type=int, default=100)
    arg_parser.add_argument("--checkpoint_src", type=str, default=None)
    args = arg_parser.parse_args()

    optimizer_warmup_iters = args.optimizer_warmup_iters if args.optimizer_warmup_iters is not None else args.iter // 50
    optimizer_cosine_cycle_iters = args.optimizer_cosine_cycle_iters if args.optimizer_cosine_cycle_iters is not None else args.iter

    assert args.optimizer_lr_min <= args.optimizer_lr_max
    assert optimizer_warmup_iters < optimizer_cosine_cycle_iters <= args.iter

    training(batch_size=args.batch_size, lr=args.optimizer_lr, )
