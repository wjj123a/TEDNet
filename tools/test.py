import argparse
import os
import os.path as osp
import sys

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description="Test a segmentor")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--work-dir", help="directory to save evaluation logs")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override settings in the config, using key=value format")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs",
                                osp.splitext(osp.basename(args.config))[0])

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.test()


if __name__ == "__main__":
    main()
