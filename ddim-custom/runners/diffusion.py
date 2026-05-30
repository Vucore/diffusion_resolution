"""
runners/diffusion.py
--------------------
Core training và sampling logic — giữ nguyên 100% logic DDIM gốc.
Thay đổi duy nhất: loại bỏ các import không cần thiết (ckpt_util)
và cập nhật get_dataset để dùng CustomImageFolderDataset.
"""

import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform

import torchvision.utils as tvu


# ── Utility ──────────────────────────────────────────────────────────────────

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    Tạo beta schedule theo kiểu được chỉ định.
    Hỗ trợ: linear (mặc định/khuyên dùng), quad, const, jsd, sigmoid.
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                        num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1,
                                   num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# ── Main Runner ───────────────────────────────────────────────────────────────

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    # ── TRAINING ─────────────────────────────────────────────────────────────

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)

        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,        # tránh batch lẻ gây lỗi antithetic sampling
        )

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        if self.args.resume_training:
            if getattr(self.args, "resume_ckpt", None) and os.path.isfile(self.args.resume_ckpt):
                ckpt_path = self.args.resume_ckpt
            else:
                ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
            logging.info(f"Resuming training từ: {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info(f"Tiếp tục từ epoch {start_epoch}, step {step}")

        logging.info(f"Bắt đầu training trên {self.device}")
        logging.info(f"Số ảnh train: {len(dataset)}")

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)   # [0,1] → [-1,1]
                e = torch.randn_like(x)              # noise target
                b = self.betas

                # Antithetic sampling của timestep (ổn định gradient hơn)
                t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, t, e, b)

                if tb_logger is not None:
                    tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"epoch: {epoch} | step: {step} | "
                    f"loss: {loss.item():.6f} | "
                    f"data_time: {data_time / (i + 1):.4f}s"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                # Lưu checkpoint
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, f"ckpt_{step}.pth"),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    logging.info(f"Đã lưu checkpoint tại step {step}")

                data_start = time.time()

                # Dừng khi đạt n_iters
                if step >= self.config.training.n_iters:
                    logging.info(f"Đạt n_iters={self.config.training.n_iters}. Dừng training.")
                    return

    # ── SAMPLING ─────────────────────────────────────────────────────────────

    def sample(self, last=True):
        model = Model(self.config)

        if getattr(self.args, "ckpt", None):
            ckpt_path = self.args.ckpt
        else:
            ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
        logging.info(f"Tải checkpoint từ: {ckpt_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.device)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)

        model.eval()
        logging.info("Bắt đầu sampling...")
        self.sample_images(model, last)

    def sample_images(self, model, last):
        """
        Sinh ảnh từ noise ngẫu nhiên và lưu vào image_folder.
        Số ảnh = config.sampling.total_samples (mặc định 64).
        - last=True: chỉ lưu ảnh cuối cùng.
        - last=False: lưu toàn bộ quá trình denoising theo từng timestep.
        """
        config = self.config
        os.makedirs(self.args.image_folder, exist_ok=True)
        total_samples = getattr(config.sampling, "total_samples", 64)
        batch_size = config.sampling.batch_size
        n_rounds = (total_samples + batch_size - 1) // batch_size
        img_id = 0

        with torch.no_grad():
            for _ in tqdm.tqdm(range(n_rounds), desc="Generating images"):
                n = min(batch_size, total_samples - img_id)
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                sample_out = self.sample_image(x, model, last)

                if torch.is_tensor(sample_out):
                    x = inverse_data_transform(config, sample_out)
                    for i in range(x.shape[0]):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{img_id:05d}.png")
                        )
                        img_id += 1
                else:
                    xs = sample_out[0]
                    x0_preds = sample_out[1] if len(sample_out) > 1 else []
                    for step_idx, x_step in enumerate(xs):
                        x_step = inverse_data_transform(config, x_step)
                        step_dir = os.path.join(self.args.image_folder, f"step_{step_idx:04d}")
                        os.makedirs(step_dir, exist_ok=True)
                        for i in range(x_step.shape[0]):
                            tvu.save_image(
                                x_step[i],
                                os.path.join(step_dir, f"{img_id + i:05d}.png"),
                            )

                        if step_idx < len(x0_preds):
                            x0_step = inverse_data_transform(config, x0_preds[step_idx])
                            x0_dir = os.path.join(
                                self.args.image_folder, f"x0_step_{step_idx:04d}"
                            )
                            os.makedirs(x0_dir, exist_ok=True)
                            for i in range(x0_step.shape[0]):
                                tvu.save_image(
                                    x0_step[i],
                                    os.path.join(x0_dir, f"{img_id + i:05d}.png"),
                                )
                    img_id += n

        logging.info(f"Đã sinh {img_id} ảnh vào: {self.args.image_folder}")

    def sample_image(self, x, model, last=True):
        """Chạy DDIM/DDPM reverse process."""
        try:
            skip = self.args.skip
        except AttributeError:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs

        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError

        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
