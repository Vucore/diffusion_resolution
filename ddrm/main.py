import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.diffusion import Diffusion
from datasets import IMAGE_EXTENSIONS

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint path (override default)"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--test_dir", type=str, default=None,
        help="Custom image folder for testing (no labels needed)"
    )
    parser.add_argument(
        "--save_test_list", action="store_true",
        help="Save a manifest for metrics reproducibility"
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--sigma_0", type=float, required=True, help="Sigma_0"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )
    parser.add_argument(
        "--etaB", type=float, default=1, help="Eta_b (before)"
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    test_images = None
    if args.test_dir is not None:
        args.test_dir = os.path.abspath(args.test_dir)
        test_images = _list_image_paths(args.test_dir)
        new_config.data.dataset = "CUSTOM"
        new_config.data.data_dir = args.test_dir
        new_config.data.random_flip = False

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    _prepare_output_dir(args.image_folder, args.ni)
    _write_yaml(os.path.join(args.image_folder, "config.yml"), config)
    _write_run_manifest(args.image_folder, args, test_images)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def _list_image_paths(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Không tìm thấy folder: {image_dir}")

    image_paths = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
    ])

    if len(image_paths) == 0:
        raise RuntimeError(
            f"Không tìm thấy ảnh nào trong thư mục: {image_dir}\n"
            f"Định dạng hỗ trợ: {IMAGE_EXTENSIONS}"
        )

    return image_paths


def _prepare_output_dir(path, no_interaction):
    if os.path.exists(path) and os.listdir(path):
        overwrite = no_interaction
        if not overwrite:
            response = input(
                f"Image folder {path} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(path)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    os.makedirs(path, exist_ok=True)


def _write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def _write_run_manifest(output_dir, args, test_images):
    manifest = {
        "cmd": " ".join(sys.argv),
        "ckpt": args.ckpt,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "eta": args.eta,
        "etaB": args.etaB,
        "deg": args.deg,
        "sigma_0": args.sigma_0,
        "subset_start": args.subset_start,
        "subset_end": args.subset_end,
    }

    if args.test_dir is not None:
        manifest["test_dir"] = args.test_dir
        manifest["test_count"] = len(test_images) if test_images else 0
        if args.save_test_list and test_images:
            manifest["test_files"] = [
                os.path.relpath(p, args.test_dir) for p in test_images
            ]

    _write_yaml(os.path.join(output_dir, "run.yml"), manifest)


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        runner.sample()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
