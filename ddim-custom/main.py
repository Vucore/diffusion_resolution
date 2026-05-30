"""
main.py
-------
Entry point để train / sample model DDIM trên custom image folder dataset.

Ví dụ training:
  python main.py --config custom.yml --doc my_run --data_dir path/to/images

Ví dụ resume:
  python main.py --config custom.yml --doc my_run --data_dir path/to/images --resume_training

Ví dụ sampling:
  python main.py --config custom.yml --doc my_run --sample --timesteps 100 --eta 0.0
"""

import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

try:
    import torch.utils.tensorboard as tb
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[Warning] TensorBoard không khả dụng. Cài bằng: pip install tensorboard")

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description="DDIM training / sampling trên custom image folder dataset"
    )

    # ── Cấu hình cơ bản ────────────────────────────────────────────────────
    parser.add_argument("--config", type=str, required=True,
                        help="Tên file config trong thư mục configs/, ví dụ: custom.yml")
    parser.add_argument("--doc", type=str, required=True,
                        help="Tên experiment (dùng làm tên folder log)")
    parser.add_argument("--exp", type=str, default="exp",
                        help="Thư mục gốc để lưu log, checkpoint, sample")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--verbose", type=str, default="info",
                        choices=["debug", "info", "warning", "critical"])

    # ── Dataset ────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, default=None,
                        help="[BẮT BUỘC] Đường dẫn tới folder chứa ảnh của bạn")

    # ── Chế độ ─────────────────────────────────────────────────────────────
    parser.add_argument("--sample", action="store_true",
                        help="Chế độ sinh ảnh (sample) thay vì training")
    parser.add_argument("--resume_training", action="store_true",
                        help="Tiếp tục training từ checkpoint ckpt.pth gần nhất")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Đường dẫn tĩnh tới file checkpoint (.pth) để train tiếp (vd: D:\\TTNT_NC\\ckpt.pth)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint dùng để sample (ưu tiên hơn ckpt.pth trong log)")
    parser.add_argument("--ni", action="store_true",
                        help="No interaction — tự động ghi đè folder (dành cho server/script)")

    # ── Sampling options ───────────────────────────────────────────────────
    parser.add_argument("-i", "--image_folder", type=str, default="samples",
                        help="Tên thư mục lưu ảnh được sinh ra")
    parser.add_argument("--sample_type", type=str, default="generalized",
                        choices=["generalized", "ddpm_noisy"],
                        help="generalized = DDIM; ddpm_noisy = DDPM gốc")
    parser.add_argument("--skip_type", type=str, default="uniform",
                        choices=["uniform", "quad"])
    parser.add_argument("--timesteps", type=int, default=100,
                        help="Số bước denoising lúc sample (nhỏ hơn = nhanh hơn, ít chi tiết hơn)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="0.0 = deterministic DDIM | 1.0 = stochastic DDPM")
    parser.add_argument("--total_samples", type=int, default=None,
                        help="Ghi đè số lượng ảnh sinh ra khi sampling")
    parser.add_argument("--sequence", action="store_true",
                        help="Lưu toàn bộ sequence denoising")
    parser.add_argument("--last", action="store_true", default=False,
                        help="Nếu True, chỉ trả về ảnh cuối cùng (x0). Nếu False, trả về toàn bộ chuỗi (sequence).")
    args = parser.parse_args()

    # ── Đường dẫn log ──────────────────────────────────────────────────────
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # ── Đọc config yaml ────────────────────────────────────────────────────
    config_path = os.path.join("configs", args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # ── Override data_dir từ CLI nếu được cung cấp ─────────────────────────
    if args.data_dir is not None:
        new_config.data.data_dir = args.data_dir
        print(f"[Config] data_dir được ghi đè từ CLI: {args.data_dir}")

    if args.total_samples is not None:
        new_config.sampling.total_samples = args.total_samples
        print(f"[Config] total_samples được ghi đè từ CLI: {args.total_samples}")

    if not hasattr(new_config.data, "data_dir") or not new_config.data.data_dir:
        if not args.sample:
            raise ValueError(
                "Bạn phải chỉ định đường dẫn dataset!\n"
                "  Cách 1: python main.py ... --data_dir /path/to/images\n"
                "  Cách 2: đặt data_dir trong configs/custom.yml"
            )

    # ── Thiết lập thư mục & logging ────────────────────────────────────────
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = args.ni or _ask_overwrite()
                if overwrite:
                    shutil.rmtree(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Dừng.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            os.makedirs(args.log_path, exist_ok=True)

        os.makedirs(tb_path, exist_ok=True)
        if HAS_TENSORBOARD:
            new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        else:
            new_config.tb_logger = None

        _setup_logger(args.verbose, args.log_path)
    else:
        _setup_logger(args.verbose)
        # Thiết lập thư mục lưu ảnh sample
        args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
        os.makedirs(args.image_folder, exist_ok=True)
        print(f"Ảnh sample sẽ được lưu tại: {args.image_folder}")

    # ── Device & Seeds ─────────────────────────────────────────────────────
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    logging.info(f"Using device: {device}")
    new_config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


def _ask_overwrite():
    response = input("Folder log đã tồn tại. Ghi đè? (Y/N): ")
    return response.strip().upper() == "Y"


def _setup_logger(verbose, log_path=None):
    level = getattr(logging, verbose.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(os.path.join(log_path, "stdout.txt")))
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    for h in handlers:
        h.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = []
    for h in handlers:
        logger.addHandler(h)
    logger.setLevel(level)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        new_value = dict2namespace(value) if isinstance(value, dict) else value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info(f"Log path: {args.log_path}")
    logging.info(f"PID: {os.getpid()}")

    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample(last=args.last)
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
