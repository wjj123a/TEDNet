from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import load_checkpoint

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tednet  # noqa: F401
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


class TensorModeWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, mode="tensor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TEDNet FPS with batch size 1 random input.")
    parser.add_argument("config", help="Config file path.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Checkpoint file. If omitted, random initialized weights are used.")
    parser.add_argument(
        "--shape",
        nargs=2,
        default=(1024, 2048),
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Input shape for FPS benchmarking.")
    parser.add_argument("--device", default="cuda:0", help="Benchmark device.")
    parser.add_argument("--warmup", default=50, type=int, help="Warmup iters.")
    parser.add_argument("--iters", default=200, type=int, help="Timed iters.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to <work_dir>/fps.json.")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1_000_000.0


def try_complexity(model: nn.Module, shape: tuple[int, int]) -> dict:
    try:
        from mmengine.analysis import get_model_complexity_info

        wrapper = TensorModeWrapper(model)
        result = get_model_complexity_info(
            wrapper,
            input_shape=(3, shape[0], shape[1]),
            show_table=False,
            show_arch=False)
        return {
            "flops": result.get("flops"),
            "flops_str": result.get("flops_str"),
            "params": result.get("params"),
            "params_str": result.get("params_str"),
        }
    except Exception as exc:
        return {"complexity_error": str(exc)}


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs",
                                osp.splitext(osp.basename(args.config))[0])

    device = torch.device(args.device if torch.cuda.is_available()
                          or not args.device.startswith("cuda") else "cpu")
    model = MODELS.build(cfg.model)
    model.to(device)
    model.eval()

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location=device)

    height, width = args.shape
    inputs = torch.randn(1, 3, height, width, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(inputs, mode="tensor")
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            model(inputs, mode="tensor")
        synchronize(device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed / args.iters * 1000.0
    fps = 1000.0 / latency_ms
    report = dict(
        config=osp.abspath(args.config),
        checkpoint=osp.abspath(args.checkpoint) if args.checkpoint else None,
        device=str(device),
        height=height,
        width=width,
        warmup=args.warmup,
        iters=args.iters,
        latency_ms=round(latency_ms, 4),
        fps=round(fps, 4),
        params_m=round(count_params(model), 4))
    report.update(try_complexity(model, (height, width)))

    output = args.output
    if output is None:
        output = osp.join(cfg.work_dir, "fps.json")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
