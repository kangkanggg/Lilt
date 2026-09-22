#!/usr/bin/env python3
"""Synthetic ResNet-50 smoke workload; not a paper benchmark or accuracy test."""
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("hp", "be"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--interval-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.interval_ms < 0:
        parser.error("steps/batch-size must be positive; interval must be nonnegative")
    import torch
    from torchvision.models import resnet50
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = resnet50(weights=None).cuda()
    data = torch.randn(args.batch_size, 3, 224, 224, device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if args.role == "be" else None
    model.train(args.role == "be")
    start = time.monotonic()
    for _ in range(args.steps):
        if optimizer is None:
            with torch.inference_mode():
                model(data)
            torch.cuda.synchronize()
            time.sleep(args.interval_ms / 1000.0)
        else:
            optimizer.zero_grad(set_to_none=True)
            model(data).square().mean().backward()
            optimizer.step()
    torch.cuda.synchronize()
    print(f"role={args.role} steps={args.steps} elapsed_s={time.monotonic()-start:.3f}")


if __name__ == "__main__":
    main()
