import os
import io
import sys
import time
import base64
import yaml
import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torchvision.transforms as transforms

# Thiết lập đường dẫn tương thích khi chạy từ thư mục gốc hoặc từ thư mục ddrm
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.diffusion import Model
from runners.diffusion import Diffusion
from datasets import data_transform, inverse_data_transform


# ─────────────────────────────────────────────────────────────────────────────
# 1. HÀM CẤU HÌNH & THIẾT BỊ
# ─────────────────────────────────────────────────────────────────────────────

def dict2namespace(config):
    namespace = type("Namespace", (), {})()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return dict2namespace(config)


def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    return devices


def load_state_dict(model, ckpt_path, device):
    """Nạp trọng số checkpoint linh hoạt và an toàn."""
    states = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(states, list):
        states = states[0]
    if isinstance(states, dict) and "state_dict" in states and isinstance(states["state_dict"], dict):
        states = states["state_dict"]
    if isinstance(states, dict) and any(k.startswith("module.") for k in states.keys()):
        stripped = {}
        for k, v in states.items():
            if k.startswith("module."):
                stripped[k[len("module.") :]] = v
            else:
                stripped[k] = v
        states = stripped
    model.load_state_dict(states, strict=True)


@st.cache_resource(show_spinner=False)
def load_model_and_config(config_path, ckpt_path, device_str):
    config = load_config(config_path)
    device = torch.device(device_str)
    model = Model(config)
    load_state_dict(model, ckpt_path, device)
    model.to(device)
    model.eval()
    return config, model


# ─────────────────────────────────────────────────────────────────────────────
# 2. HÀM HỖ TRỢ XỬ LÝ MASK VÀ SVD TRONG DDRM
# ─────────────────────────────────────────────────────────────────────────────

def _load_inp_mask(mask_path, image_size, device):
    loaded = np.load(mask_path)
    if loaded.ndim == 3:
        if loaded.shape[0] in (1, 3) and loaded.shape[0] != loaded.shape[1]:
            loaded = loaded.transpose(1, 2, 0)
        loaded = loaded[:, :, 0]
    if loaded.max() <= 1:
        loaded = (loaded * 255).astype(np.uint8)
    else:
        loaded = loaded.astype(np.uint8)
    mask_img = Image.fromarray(loaded)
    if mask_img.size != (image_size, image_size):
        mask_img = mask_img.resize((image_size, image_size), resample=Image.NEAREST)
    mask = (np.array(mask_img) > 0).astype(np.uint8)
    return torch.from_numpy(mask).to(device)


def _mask_to_missing_indices(mask, channels, device):
    if mask.dim() == 2:
        mask = mask.reshape(-1)
    mask = mask.to(device)
    missing_r = torch.nonzero(mask == 0).long().reshape(-1) * channels
    missing = [missing_r + ch for ch in range(channels)]
    return torch.cat(missing, dim=0)


def build_degradation(deg, config, device, inp_mask=None):
    channels = config.data.channels
    image_size = config.data.image_size

    if deg == "deno":
        from functions.svd_replacement import Denoising
        return Denoising(channels, image_size, device)

    elif deg.startswith("sr"):
        ratio = int(deg.replace("sr", ""))
        from functions.svd_replacement import SuperResolution
        return SuperResolution(channels, image_size, ratio, device)

    elif deg == "deblur_uni":
        from functions.svd_replacement import Deblurring
        kernel = torch.tensor([1.0 / 9.0] * 9, device=device)
        return Deblurring(kernel, channels, image_size, device)

    elif deg == "deblur_gauss":
        from functions.svd_replacement import Deblurring
        sigma = 10
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel = torch.stack([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).flatten()
        return Deblurring(kernel / kernel.sum(), channels, image_size, device)

    elif deg == "deblur_aniso":
        from functions.svd_replacement import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel2 = torch.stack([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).flatten()
        sigma = 1
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel1 = torch.stack([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).flatten()
        return Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), channels, image_size, device)

    elif deg == "color":
        from functions.svd_replacement import Colorization
        return Colorization(image_size, device)

    elif deg.startswith("inp"):
        from functions.svd_replacement import Inpainting
        if deg == "inp_lolcat":
            mask_path = os.path.join(CURRENT_DIR, "inp_masks", "lolcat_extra.npy")
            mask = _load_inp_mask(mask_path, image_size, device) if os.path.exists(mask_path) else None
        elif deg == "inp_lorem":
            mask_path = os.path.join(CURRENT_DIR, "inp_masks", "lorem3.npy")
            mask = _load_inp_mask(mask_path, image_size, device) if os.path.exists(mask_path) else None
        elif inp_mask is not None:
            mask = inp_mask.to(device)
        else:
            mask = None

        if mask is None:
            # Luôn tạo mask ngẫu nhiên cố định trên CPU với generator seed 1234 để đồng bộ với bộ ảnh Test_Degraded
            g = torch.Generator(device="cpu")
            g.manual_seed(1234)
            missing_r = torch.randperm(image_size ** 2, generator=g, device="cpu").to(device)
            missing_r = missing_r[: image_size ** 2 // 2].long() * channels
            missing = [missing_r + ch for ch in range(channels)]
            missing = torch.cat(missing, dim=0)
        else:
            missing = _mask_to_missing_indices(mask, channels, device)

        return Inpainting(channels, image_size, missing, device)

    raise ValueError(f"Tác vụ không hợp lệ: {deg}")


def infer_inp_mask_from_image(input_tensor):
    """
    Tự động suy luận mặt nạ vùng bị che đen (r=g=b≈0) từ bức ảnh đã bị che khuyết.
    Trả về mask nhị phân (H, W) với 1 là pixel còn nguyên và 0 là pixel bị che đen.
    """
    # input_tensor: (1, 3, H, W) trong khoảng [0, 1]
    is_black = (input_tensor.squeeze(0).max(dim=0).values < (2.0 / 255.0))
    missing_ratio = is_black.float().mean().item()
    # Nếu tỷ lệ điểm đen nằm trong khoảng hợp lý (5% -> 85%) thì dùng mask phát hiện được
    if 0.05 <= missing_ratio <= 0.85:
        mask = (~is_black).to(torch.uint8)
        return mask
    return None


def prepare_measurement(input_tensor, config, deg, device, H_funcs, sigma_0, simulate=True):
    """
    Chuẩn bị measurement y_0 và ảnh suy thoái để hiển thị so sánh.
    """
    meas = data_transform(config, input_tensor.to(device))  # Chuyển về khoảng [-1, 1]

    if simulate:
        # Giả lập quá trình làm suy thoái ảnh qua toán tử H
        y_0 = H_funcs.H(meas)
        if sigma_0 > 0:
            if deg.startswith("sr"):
                ratio = int(deg.replace("sr", ""))
                y_0 = y_0 + (sigma_0 / ratio) * torch.randn_like(y_0)
            else:
                y_0 = y_0 + sigma_0 * torch.randn_like(y_0)

        pinv_y_0 = H_funcs.H_pinv(y_0).view(1, config.data.channels, config.data.image_size, config.data.image_size)
        if deg.startswith("deblur"):
            pinv_y_0 = y_0.view(1, config.data.channels, config.data.image_size, config.data.image_size)
        elif deg.startswith("inp"):
            ones_meas = H_funcs.H(torch.ones_like(pinv_y_0))
            pinv_y_0 += H_funcs.H_pinv(ones_meas).reshape(*pinv_y_0.shape) - 1.0

        degraded_tensor = inverse_data_transform(config, pinv_y_0).clamp(0, 1).squeeze(0).cpu()
        degraded_pil = transforms.ToPILImage()(degraded_tensor)
    else:
        # Ảnh đưa vào đã bị suy thoái sẵn từ ngoài (không thêm nhiễu giả lập lần 2)
        if deg.startswith("sr"):
            y_0 = H_funcs.H(meas)
            pinv_y_0 = meas
        elif deg.startswith("inp"):
            y_0 = H_funcs.H(meas)
            pinv_y_0 = meas
        elif deg == "color":
            # Colorization.Vt yêu cầu đầu vào 3 kênh (1, 3, H, W)
            if meas.shape[1] == 1:
                meas = meas.repeat(1, 3, 1, 1)
            else:
                gray = meas.mean(dim=1, keepdim=True)
                meas = gray.repeat(1, 3, 1, 1)
            y_0 = H_funcs.H(meas)
            pinv_y_0 = meas
        else:
            # deno, deblur_gauss, deblur_uni, deblur_aniso:
            # Trong Deblurring, H(x) trả về trực tiếp ảnh mờ trong miền không gian (1, C*H*W)
            y_0 = meas.reshape(1, -1)
            pinv_y_0 = meas

        degraded_tensor = inverse_data_transform(config, pinv_y_0).clamp(0, 1).squeeze(0).cpu()
        degraded_pil = transforms.ToPILImage()(degraded_tensor)

    return y_0, degraded_pil


# ─────────────────────────────────────────────────────────────────────────────
# 3. THUẬT TOÁN SINH ẢNH DDIM & DDPM TỪ NOISE
# ─────────────────────────────────────────────────────────────────────────────

def sample_ddim(
    model,
    config,
    timesteps=20,
    eta=0.0,
    total_samples=4,
    seed=1234,
    device=None,
    progress_bar=None,
    status_text=None,
    live_preview=None,
    method="ddim",
):
    """
    Sinh ảnh mới từ nhiễu Gaussian N(0, I) bằng DDIM hoặc DDPM nguyên bản:
    - method="ddim": Lấy mẫu phi Markov (Non-Markovian), nhảy bước trực tiếp tới next_t -> chỉ cần 20 bước.
    - method="ddpm": Chuỗi Markov bước đơn cố định (Ho et al., 2020) với hệ số beta_t gốc của lịch T=1000
                     -> cần đủ T=1000 bước mới khử sạch nhiễu; nếu chạy ít bước (20 bước) ảnh sẽ còn đầy nhiễu.
    Trả về (final_images, step_history, elapsed_sec).
    """
    t_start = time.perf_counter()

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    b_start = config.diffusion.beta_start
    b_end = config.diffusion.beta_end
    num_timesteps = config.diffusion.num_diffusion_timesteps
    betas = np.linspace(b_start, b_end, num_timesteps, dtype=np.float64)
    b = torch.from_numpy(betas).float().to(device)

    skip = max(1, num_timesteps // timesteps)
    seq = list(range(0, num_timesteps, skip))
    if len(seq) > timesteps:
        seq = seq[:timesteps]
    seq_next = [-1] + list(seq[:-1])

    # Khởi tạo nhiễu Gaussian ngẫu nhiên ban đầu (cùng Seed luôn tạo ra chính xác cùng x_T)
    x = torch.randn(
        total_samples,
        config.data.channels,
        config.data.image_size,
        config.data.image_size,
        device=device,
    )

    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    # Ghi lại lịch sử từng bước để phục vụ xem lại trên thanh trượt 60 FPS
    step_history = []
    init_imgs = inverse_data_transform(config, x).clamp(0.0, 1.0).cpu()
    init_pil_list = [transforms.ToPILImage()(init_imgs[idx]) for idx in range(total_samples)]
    step_history.append({
        "step": 0,
        "timestep": num_timesteps,
        "title": "Bước 0: Nhiễu ban đầu",
        "desc": "Mảng nhiễu Gaussian ngẫu nhiên N(0, I)",
        "images": init_pil_list,
    })

    method_upper = method.upper()
    with torch.no_grad():
        n = x.size(0)
        total_steps = len(seq)
        # Giới hạn tối đa ~50 khung hình lịch sử để thanh trượt 60 FPS luôn mượt kể cả khi T=1000
        record_stride = max(1, total_steps // 50)
        # Cập nhật UI vừa phải khi chạy 1000 bước để đạt tốc độ phần cứng tối đa
        ui_stride = max(1, total_steps // 25)

        for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t_curr = i
            t_next = max(j, 0)
            current_step_num = step_idx + 1
            percent = int(current_step_num / total_steps * 100)
            is_last_step = (current_step_num == total_steps)

            t = torch.full((n,), i, device=device, dtype=torch.long)
            next_t = torch.full((n,), j, device=device, dtype=torch.long)

            at = compute_alpha(b, t)
            et = model(x, t.float() if method == "ddpm" else t)
            if et.shape[1] == 6:
                et = et[:, :3]

            x0_t = (x - et * (1.0 - at).sqrt()) / at.sqrt()
            x0_t = torch.clamp(x0_t, -1.0, 1.0)

            if method == "ddpm":
                # DDPM nguyên bản (Ho et al., 2020):
                # Chuỗi Markov bước đơn cố định với hệ số beta_t gốc của lịch T=1000
                beta_t = b.index_select(0, t).view(-1, 1, 1, 1)
                alpha_t = 1.0 - beta_t
                at_prev_markov = at / alpha_t  # alpha_bar_{t-1} của bước đơn Markov

                mean = (
                    (at_prev_markov.sqrt() * beta_t) * x0_t
                    + (alpha_t.sqrt() * (1.0 - at_prev_markov)) * x
                ) / (1.0 - at)

                noise = torch.randn_like(x)
                mask = (t > 0).float().view(-1, 1, 1, 1)
                x = mean + mask * beta_t.sqrt() * noise
            else:
                # DDIM (Song et al., 2021):
                # Quá trình phi Markov (Non-Markovian), nhảy bước trực tiếp tới at_next
                at_next = compute_alpha(b, next_t)
                c1 = eta * ((1.0 - at / at_next) * (1.0 - at_next) / (1.0 - at)).clamp_min(0.0).sqrt()
                c2 = ((1.0 - at_next) - c1 ** 2).clamp_min(0.0).sqrt()
                noise = torch.randn_like(x) if eta > 0.0 else 0.0
                x = at_next.sqrt() * x0_t + c1 * noise + c2 * et

            should_record = (current_step_num % record_stride == 0) or is_last_step
            should_update_ui = (current_step_num % ui_stride == 0) or is_last_step

            if should_record or should_update_ui:
                curr_imgs = inverse_data_transform(config, x).clamp(0.0, 1.0).cpu()
                curr_pil_list = [transforms.ToPILImage()(curr_imgs[idx]) for idx in range(total_samples)]

                if should_record:
                    step_history.append({
                        "step": current_step_num,
                        "timestep": t_next,
                        "title": f"Bước {current_step_num}/{total_steps}",
                        "desc": f"Timestep {t_curr} → {t_next} ({percent}%)",
                        "images": curr_pil_list,
                    })

                if should_update_ui:
                    if status_text is not None:
                        status_text.text(
                            f"⏳ [{method_upper}] Đang khử nhiễu bước {current_step_num}/{total_steps} "
                            f"(Timestep {t_curr} → {t_next} • {percent}%)..."
                        )
                    if progress_bar is not None:
                        progress_bar.progress(current_step_num / total_steps)
                    if live_preview is not None and len(curr_pil_list) > 0:
                        live_preview.image(
                            curr_pil_list[0].resize((160, 160), Image.NEAREST),
                            caption=f"⚡ [{method_upper}] Bước {current_step_num}/{total_steps} ({percent}%)",
                            width=160,
                        )

    elapsed_sec = time.perf_counter() - t_start
    final_images = step_history[-1]["images"]
    return final_images, step_history, elapsed_sec


# ─────────────────────────────────────────────────────────────────────────────
# 4. GIAO DIỆN THANH TRƯỢT SO SÁNH TRỰC QUAN (BEFORE / AFTER)
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_compare_slider(left_img, right_img, width, height, key, label_left="Ảnh suy thoái", label_right="DDRM Phục hồi"):
    left_b64 = _pil_to_base64(left_img)
    right_b64 = _pil_to_base64(right_img)
    html = f"""
<style>
.comp-wrapper {{
    width: {width}px;
    margin: 8px auto 0 auto;
    user-select: none;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
.comp-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    width: 100%;
}}
.comp-header-badge {{
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}}
.badge-left-hdr {{
    background: #FEE2E2;
    color: #DC2626;
    border: 1px solid #FECACA;
}}
.badge-right-hdr {{
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #A7F3D0;
}}

#comp-{key} {{
    position: relative;
    height: {height}px;
    width: {width}px;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    margin: 0 auto;
}}
#comp-{key} .img-comp-img {{
    position: absolute;
    width: {width}px;
    height: {height}px;
    overflow: hidden;
}}
#comp-{key} .img-comp-base {{
    z-index: 1;
}}
#comp-{key} .img-comp-overlay {{
    z-index: 2;
}}
#comp-{key} img {{
    display: block;
    width: {width}px;
    height: {height}px;
    image-rendering: pixelated;
}}

/* Corner badges inside images - đặt ở góc trên, không che mặt và tự ẩn theo con trượt */
.in-img-badge {{
    position: absolute;
    top: 14px;
    padding: 6px 14px;
    font-size: 11.5px;
    font-weight: 700;
    border-radius: 8px;
    color: white;
    backdrop-filter: blur(8px);
    pointer-events: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    transition: opacity 0.12s ease-out;
}}
.badge-left-in {{
    left: 14px;
    background: rgba(220, 38, 38, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.3);
}}
.badge-right-in {{
    right: 14px;
    background: rgba(16, 185, 129, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.3);
}}

#comp-{key} .img-comp-slider {{
    position: absolute;
    z-index: 15;
    cursor: ew-resize;
    width: 3px;
    background: #FFFFFF;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
    height: {height}px;
    top: 0;
    left: calc(50% - 1.5px);
}}
#comp-{key} .img-comp-handle {{
    position: absolute;
    top: calc(50% - 20px);
    left: -19px;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #4F46E5;
    border: 3px solid #FFFFFF;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 16px;
    font-weight: bold;
}}
</style>

<div class="comp-wrapper">
    <!-- Thanh nhãn định vị rõ ràng BÊN TRÁI vs BÊN PHẢI phía trên ảnh (không bị co chữ) -->
    <div class="comp-header">
        <div class="comp-header-badge badge-left-hdr">🔴 Bên trái: {label_left}</div>
        <div class="comp-header-badge badge-right-hdr">🟢 Bên phải: {label_right}</div>
    </div>

    <!-- Khung ảnh so sánh -->
    <div id="comp-{key}">
        <!-- Lớp nền (BÊN PHẢI): Ảnh đã phục hồi -->
        <div class="img-comp-img img-comp-base">
            <img src="data:image/png;base64,{right_b64}" />
            <div class="in-img-badge badge-right-in" id="badge-right-{key}">{label_right}</div>
        </div>

        <!-- Lớp phủ (BÊN TRÁI): Ảnh suy thoái, co giãn theo thanh trượt -->
        <div class="img-comp-img img-comp-overlay" id="overlay-{key}">
            <img src="data:image/png;base64,{left_b64}" />
            <div class="in-img-badge badge-left-in" id="badge-left-{key}">{label_left}</div>
        </div>

        <!-- Con trượt kéo -->
        <div class="img-comp-slider" id="slider-{key}">
            <div class="img-comp-handle">⬌</div>
        </div>
    </div>
</div>

<script>
(function() {{
    var container = document.getElementById("comp-{key}");
    var overlay = document.getElementById("overlay-{key}");
    var slider = document.getElementById("slider-{key}");
    var badgeLeft = document.getElementById("badge-left-{key}");
    var badgeRight = document.getElementById("badge-right-{key}");
    var width = {width};
    var clicked = false;

    function setPos(x) {{
        x = Math.max(0, Math.min(x, width));
        overlay.style.width = x + "px";
        slider.style.left = (x - 1.5) + "px";

        // Tự động mờ dần và ẩn nhãn khi con trượt kéo sát mép
        if (badgeLeft) {{
            badgeLeft.style.opacity = Math.max(0, Math.min(1, (x - 25) / 50));
        }}
        if (badgeRight) {{
            badgeRight.style.opacity = Math.max(0, Math.min(1, (width - x - 25) / 50));
        }}
    }}

    function getCursorPos(e) {{
        var rect = container.getBoundingClientRect();
        var x = 0;
        if (e.touches && e.touches.length > 0) {{
            x = e.touches[0].clientX - rect.left;
        }} else {{
            x = e.clientX - rect.left;
        }}
        return x;
    }}

    function slideMove(e) {{
        if (!clicked) return;
        setPos(getCursorPos(e));
    }}

    function slideReady(e) {{
        e.preventDefault();
        clicked = true;
        window.addEventListener("mousemove", slideMove);
        window.addEventListener("touchmove", slideMove);
    }}

    function slideFinish() {{
        clicked = false;
        window.removeEventListener("mousemove", slideMove);
        window.removeEventListener("touchmove", slideFinish);
    }}

    // Khởi tạo vị trí 50% ở chính giữa
    setPos(width / 2);

    slider.addEventListener("mousedown", slideReady);
    slider.addEventListener("touchstart", slideReady);
    window.addEventListener("mouseup", slideFinish);
    window.addEventListener("touchend", slideFinish);
}})();
</script>
"""
    components.html(html, height=height + 80, width=width + 30)


def render_ddim_step_scrubber(
    step_history,
    display_w,
    key="ddim_scrub",
    label_prefix="⏩ Tiến trình:",
    accent_color="#4F46E5",
    badge_bg="#EEF2FF",
    badge_text="#4338CA",
    badge_border="#C7D2FE",
):
    """
    Thanh trượt xem từng bước khử nhiễu DDIM / DDPM với cập nhật trực tiếp tại thời gian thực (Client-side 60 FPS).
    Hỗ trợ cả T=20 bước lẫn T=1000 bước mượt mà thông qua ánh xạ chỉ số bước thực (step_numbers).
    """
    import json

    num_samples = len(step_history[0]["images"])
    max_idx = len(step_history) - 1
    step_numbers = [int(step_data.get("step", idx)) for idx, step_data in enumerate(step_history)]
    total_timesteps = step_numbers[-1] if step_numbers[-1] > 0 else max_idx

    frames = []
    for s_idx in range(num_samples):
        s_frames = []
        for step_data in step_history:
            pil_img = step_data["images"][s_idx]
            s_frames.append(_pil_to_base64(pil_img))
        frames.append(s_frames)

    frames_json = json.dumps(frames)
    step_numbers_json = json.dumps(step_numbers)
    card_w = min(display_w, 220) if num_samples > 1 else display_w
    total_height = card_w + 135

    cards_html = ""
    for s_idx in range(num_samples):
        init_img_b64 = frames[s_idx][-1]
        cards_html += f"""
        <div class="scrub-card-{key}" style="width: {card_w}px;">
            <div class="scrub-card-title-{key}">Ảnh #{s_idx + 1}</div>
            <img id="scrub-img-{key}-{s_idx}" class="scrub-img-{key}" src="data:image/png;base64,{init_img_b64}" style="width: {card_w}px; height: {card_w}px;" />
        </div>
        """

    html = f"""
<style>
.scrub-container-{key} {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    user-select: none;
    margin: 6px auto 0 auto;
    width: 100%;
}}
.scrub-bar-box-{key} {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}}
.scrub-slider-label-{key} {{
    font-size: 13px;
    font-weight: 700;
    color: #1E293B;
    white-space: nowrap;
}}
.scrub-range-{key} {{
    flex: 1;
    height: 8px;
    border-radius: 4px;
    background: #E2E8F0;
    outline: none;
    -webkit-appearance: none;
    appearance: none;
    cursor: pointer;
}}
.scrub-range-{key}::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: {accent_color};
    border: 3px solid #FFFFFF;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    cursor: pointer;
    transition: transform 0.1s ease;
}}
.scrub-range-{key}::-webkit-slider-thumb:hover {{
    transform: scale(1.15);
}}
.scrub-range-{key}::-moz-range-thumb {{
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: {accent_color};
    border: 3px solid #FFFFFF;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    cursor: pointer;
}}
.scrub-badge-{key} {{
    background: {badge_bg};
    color: {badge_text};
    border: 1px solid {badge_border};
    font-size: 12.5px;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 18px;
    white-space: nowrap;
    min-width: 135px;
    text-align: center;
}}
.scrub-grid-{key} {{
    display: flex;
    justify-content: center;
    gap: 14px;
    flex-wrap: wrap;
}}
.scrub-card-{key} {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}}
.scrub-card-title-{key} {{
    font-size: 13px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 8px;
}}
.scrub-img-{key} {{
    display: block;
    image-rendering: pixelated;
    border-radius: 8px;
    margin: 0 auto;
}}
</style>

<div class="scrub-container-{key}">
    <div class="scrub-bar-box-{key}">
        <div class="scrub-slider-label-{key}">{label_prefix}</div>
        <input 
            type="range" 
            class="scrub-range-{key}" 
            id="slider-{key}" 
            min="0" 
            max="{max_idx}" 
            value="{max_idx}" 
            step="1"
        />
        <div class="scrub-badge-{key}" id="badge-{key}">Bước {total_timesteps}/{total_timesteps} (100%)</div>
    </div>

    <div class="scrub-grid-{key}">
        {cards_html}
    </div>
</div>

<script>
(function() {{
    var framesData = {frames_json};
    var stepNumbers = {step_numbers_json};
    var maxIdx = {max_idx};
    var totalSteps = {total_timesteps};
    var numSamples = {num_samples};
    var slider = document.getElementById("slider-{key}");
    var badge = document.getElementById("badge-{key}");

    function updateStep(val) {{
        var idx = parseInt(val);
        var realStep = stepNumbers[idx] !== undefined ? stepNumbers[idx] : idx;
        if (badge) {{
            if (realStep === 0) {{
                badge.innerText = "Bước 0: Nhiễu 100%";
            }} else if (idx === maxIdx) {{
                badge.innerText = "Bước " + realStep + "/" + totalSteps + " (100%)";
            }} else {{
                var pct = Math.round((realStep / totalSteps) * 100);
                badge.innerText = "Bước " + realStep + "/" + totalSteps + " (" + pct + "%)";
            }}
        }}

        for (var s = 0; s < numSamples; s++) {{
            var imgEl = document.getElementById("scrub-img-{key}-" + s);
            if (imgEl && framesData[s] && framesData[s][idx]) {{
                imgEl.src = "data:image/png;base64," + framesData[s][idx];
            }}
        }}
    }}

    if (slider) {{
        slider.addEventListener("input", function() {{
            updateStep(this.value);
        }});
        slider.addEventListener("change", function() {{
            updateStep(this.value);
        }});
    }}
}})();
</script>
"""
    components.html(html, height=total_height + 30, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DANH SÁCH TÁC VỤ PHỤC HỒI DDRM (DỄ HIỂU, KHÔNG DÙNG TỪ KHÓ)
# ─────────────────────────────────────────────────────────────────────────────

RESTORATION_TASKS = {
    "sr4": {
        "name": "🔍 Tăng nét 4x (Super-Resolution)",
        "desc": "Tái tạo chi tiết sắc nét và nâng độ phân giải gấp 4 lần.",
        "deg": "sr4",
        "sigma_0": 0.05,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "sr2": {
        "name": "🔍 Tăng nét 2x (Super-Resolution)",
        "desc": "Làm nét và nâng độ phân giải gấp 2 lần.",
        "deg": "sr2",
        "sigma_0": 0.05,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "deno": {
        "name": "🧹 Khử hạt nhiễu (Denoising)",
        "desc": "Lọc sạch các hạt nhiễu ngẫu nhiên, giúp ảnh mịn và trong trẻo.",
        "deg": "deno",
        "sigma_0": 0.05,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "deblur_gauss": {
        "name": "🌫️ Khử mờ ống kính (Gaussian Blur)",
        "desc": "Khắc phục ảnh bị mất nét hoặc out nét do tiêu cự camera.",
        "deg": "deblur_gauss",
        "sigma_0": 0.0,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "deblur_uni": {
        "name": "📷 Khử mờ đều (Uniform Blur)",
        "desc": "Khử hiện tượng nhòe mờ đồng đều trên toàn bộ khung hình.",
        "deg": "deblur_uni",
        "sigma_0": 0.0,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "deblur_aniso": {
        "name": "🌊 Khử mờ rung tay (Motion Blur)",
        "desc": "Khử các vệt nhòe do rung tay hoặc đối tượng chuyển động khi chụp.",
        "deg": "deblur_aniso",
        "sigma_0": 0.0,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "inp": {
        "name": "🩹 Vá ảnh che khuyết (Inpainting)",
        "desc": "Tự động tái tạo và phủ kín các vùng ảnh bị mất hoặc bị che đen.",
        "deg": "inp",
        "sigma_0": 0.0,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "inp_lolcat": {
        "name": "🐱 Vá chữ nghệ thuật (Lolcat Mask)",
        "desc": "Xóa các dòng chữ che khuất và khôi phục khuôn mặt phía sau.",
        "deg": "inp_lolcat",
        "sigma_0": 0.0,
        "eta": 0.85,
        "etaB": 1.0,
    },
    "color": {
        "name": "🎨 Tô màu ảnh đen trắng (Colorization)",
        "desc": "Nhận diện đường nét và tô màu sắc tự nhiên cho ảnh xám.",
        "deg": "color",
        "sigma_0": 0.0,
        "eta": 0.85,
        "etaB": 1.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 6. GIAO DIỆN CHÍNH (STREAMLIT APP)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Diffusion Studio • DDRM & DDIM",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── PHONG CÁCH GIAO DIỆN HIỆN ĐẠI, GỌN GÀNG, CAO CẤP ───────────────────
    st.markdown(
        """
        <style>
        /* Nền và phông chữ chung */
        .stApp {
            background-color: #F8FAFC;
            color: #0F172A;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Header tinh tế, gọn gàng */
        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 16px 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
        }
        .app-header-left {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .app-title {
            font-size: 22px;
            font-weight: 800;
            color: #0F172A;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: -0.3px;
        }
        .app-subtitle {
            font-size: 13.5px;
            color: #64748B;
            margin: 0;
        }
        .app-header-badges {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .header-pill {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .pill-device {
            background: #EFF6FF;
            color: #2563EB;
            border: 1px solid #BFDBFE;
        }
        .pill-model {
            background: #F1F5F9;
            color: #475569;
            border: 1px solid #E2E8F0;
        }

        /* Khung hiển thị ảnh */
        .img-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 12px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
            margin-bottom: 10px;
        }
        .img-card-title {
            font-size: 13.5px;
            font-weight: 700;
            color: #334155;
            margin-bottom: 8px;
        }

        /* Style cho Tabs Streamlit mượt mà, phẳng, hiện đại */
        div[data-baseweb="tab-list"] {
            gap: 8px;
            background: #F1F5F9;
            padding: 6px;
            border-radius: 12px;
            margin-bottom: 18px;
        }
        div[data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: 600;
            font-size: 14px;
            color: #475569;
            border: none !important;
            background: transparent;
            transition: all 0.2s ease;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #FFFFFF !important;
            color: #4F46E5 !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        }

        /* Nút bấm chính (Primary Button) */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%);
            border: none;
            color: #FFFFFF;
            padding: 10px 24px;
            font-weight: 700;
            font-size: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            transition: all 0.2s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── ĐƯỜNG DẪN MẶC ĐỊNH ──────────────────────────────────────────────────
    default_ckpt = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ckpt_708000.pth"))
    default_config = os.path.abspath(os.path.join(CURRENT_DIR, "configs", "custom.yml"))
    test_dir = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Test"))

    # ── SIDEBAR CÀI ĐẶT ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt hệ thống")
        available_devices = get_available_devices()
        device_choice = st.selectbox(
            "Thiết bị tính toán (Device):",
            available_devices,
            index=0,
            key="sidebar_device_select",
        )

        zoom_label = st.selectbox(
            "Kích thước hiển thị:",
            ["2x (128px)", "4x (256px)", "6x (384px)", "Gốc 1x (64px)"],
            index=1,
            key="sidebar_zoom_select",
        )
        zoom_factor = 4 if "4x" in zoom_label else (2 if "2x" in zoom_label else (6 if "6x" in zoom_label else 1))

        with st.expander("📁 Tệp Checkpoint & Config", expanded=False):
            ckpt_path = st.text_input("Đường dẫn Model (.pth):", value=default_ckpt, key="sidebar_ckpt_path")
            config_path = st.text_input("Đường dẫn Config (.yml):", value=default_config, key="sidebar_config_path")

    # Kiểm tra tồn tại file model và config
    if not os.path.isfile(ckpt_path):
        st.error(f"❌ Không tìm thấy file model checkpoint tại: `{ckpt_path}`")
        return

    if not os.path.isfile(config_path):
        st.error(f"❌ Không tìm thấy file config tại: `{config_path}`")
        return

    # Tải model vào bộ nhớ cache
    with st.spinner("Đang nạp mô hình vào bộ nhớ..."):
        try:
            config, model = load_model_and_config(config_path, ckpt_path, device_choice)
        except Exception as e:
            st.error(f"Lỗi khi nạp mô hình: {e}")
            return

    # ── HEADER GỌN GÀNG, TINH TẾ ─────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-left">
                <h1 class="app-title">✨ AI Diffusion Studio</h1>
                <p class="app-subtitle">Mô hình khuếch tán: Khôi phục ảnh (DDRM) & Sinh ảnh khuôn mặt (DDIM / DDPM)</p>
            </div>
            <div class="app-header-badges">
                <span class="header-pill pill-model">📐 {config.data.image_size}×{config.data.image_size} px</span>
                <span class="header-pill pill-device">⚡ {device_choice.upper()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── TABS CHỨC NĂNG ───────────────────────────────────────────────────────
    tab_restore, tab_generate, tab_guide = st.tabs([
        "🛠️ Phục hồi ảnh (DDRM)",
        "🎨 Sinh ảnh: DDIM vs DDPM",
        "📖 Hướng dẫn & Giới thiệu",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: PHỤC HỒI ẢNH (DDRM)
    # ═════════════════════════════════════════════════════════════════════════
    with tab_restore:
        col_input, col_settings = st.columns([1, 1], gap="large")

        input_pil = None
        suggested_task = None
        degraded_base_dir = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Test_Degraded"))

        with col_input:
            st.markdown("#### 1. Chọn ảnh đầu vào")
            source_options = ["🖼️ Ảnh mẫu (Test)", "📁 Tải ảnh từ máy tính"]
            if os.path.isdir(degraded_base_dir):
                source_options.insert(1, "📦 Bộ ảnh hư tổn (Test_Degraded)")

            input_source = st.radio(
                "Nguồn ảnh:",
                source_options,
                horizontal=True,
                key="ddrm_input_source_radio",
            )

            if "Test_Degraded" in input_source:
                if os.path.isdir(degraded_base_dir):
                    subfolders = sorted([f for f in os.listdir(degraded_base_dir) if os.path.isdir(os.path.join(degraded_base_dir, f))])
                    col_subf, col_subimg = st.columns(2)
                    with col_subf:
                        selected_folder = st.selectbox("Thư mục lỗi:", subfolders, index=0, key="ddrm_deg_folder_select")
                    folder_path = os.path.join(degraded_base_dir, selected_folder)
                    deg_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                    with col_subimg:
                        selected_deg_file = st.selectbox("Chọn ảnh:", deg_files, index=0, key="ddrm_deg_file_select")

                    input_pil = Image.open(os.path.join(folder_path, selected_deg_file))

                    folder_to_deg = {
                        "01_denoise_hat": "deno",
                        "02_sieu_phan_giai_sr2": "sr2",
                        "03_sieu_phan_giai_sr4": "sr4",
                        "04_mo_gaussian": "deblur_gauss",
                        "05_mo_deu_uniform": "deblur_uni",
                        "06_mo_rung_tay_motion": "deblur_aniso",
                        "07_va_anh_inpainting": "inp",
                        "08_va_chu_lolcat": "inp_lolcat",
                        "09_den_trang_color": "color",
                    }
                    suggested_task = folder_to_deg.get(selected_folder, None)

            elif "Ảnh mẫu" in input_source:
                if os.path.isdir(test_dir):
                    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
                    all_test_imgs = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)])
                    if all_test_imgs:
                        sample_pool = all_test_imgs[:100]
                        col_select, col_rand = st.columns([3, 2])
                        with col_rand:
                            st.write("")
                            st.write("")
                            if st.button("🎲 Đổi ảnh", key="ddrm_rand_img_btn"):
                                st.session_state["ddrm_test_img_select"] = str(np.random.choice(sample_pool))
                        with col_select:
                            selected_file = st.selectbox(
                                "Chọn ảnh mẫu:",
                                sample_pool,
                                index=0,
                                key="ddrm_test_img_select",
                            )

                        img_full_path = os.path.join(test_dir, selected_file)
                        input_pil = Image.open(img_full_path)
                    else:
                        st.warning("Thư mục Test không có file ảnh nào.")
                else:
                    st.warning(f"Không tìm thấy thư mục Test tại: `{test_dir}`")

            else:
                uploaded = st.file_uploader(
                    "Tải lên ảnh bất kỳ:",
                    type=["png", "jpg", "jpeg", "webp", "bmp"],
                    key="ddrm_file_uploader",
                )
                if uploaded is not None:
                    input_pil = Image.open(uploaded)

            if input_pil is not None:
                image_size = config.data.image_size
                input_pil = input_pil.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
                display_input = input_pil.resize((image_size * zoom_factor, image_size * zoom_factor), Image.NEAREST)

                preview_title = "Ảnh hư tổn đầu vào" if "Test_Degraded" in input_source else "Ảnh đầu vào"
                st.markdown(
                    f"""
                    <div class="img-card">
                        <div class="img-card-title">{preview_title} ({image_size}×{image_size} px)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(display_input, use_container_width=False, width=image_size * zoom_factor)

        with col_settings:
            st.markdown("#### 2. Thiết lập phục hồi")
            task_keys = list(RESTORATION_TASKS.keys())
            default_task_idx = 0
            if suggested_task and suggested_task in task_keys:
                default_task_idx = task_keys.index(suggested_task)

            task_select_key = f"task_sel_{input_source}_{suggested_task or 'default'}"
            task_key = st.selectbox(
                "Mục đích xử lý ảnh:",
                task_keys,
                format_func=lambda k: RESTORATION_TASKS[k]["name"],
                index=default_task_idx,
                key=task_select_key,
            )
            selected_task = RESTORATION_TASKS[task_key]
            st.caption(f"💡 *{selected_task['desc']}*")

            default_sim_idx = 1 if "Test_Degraded" in input_source else 0
            sim_choice = st.radio(
                "Chế độ thực hiện:",
                [
                    "Mô phỏng suy thoái (Tạo hư tổn từ ảnh mẫu & Khôi phục)",
                    "Khôi phục trực tiếp (Ảnh đã bị hư tổn sẵn)",
                ],
                index=default_sim_idx,
                key=f"sim_mode_{input_source}",
            )
            is_simulate = "Mô phỏng" in sim_choice

            with st.expander("⚙️ Tinh chỉnh nâng cao (Tùy chọn)", expanded=False):
                timesteps = st.slider(
                    "Số bước khử nhiễu (Timesteps):",
                    min_value=10,
                    max_value=100,
                    value=20,
                    step=5,
                    key="ddrm_timesteps_slider",
                )
                seed_val = st.number_input(
                    "Mã số ngẫu nhiên (Seed):",
                    min_value=0,
                    max_value=999999,
                    value=1234,
                    step=1,
                    key="ddrm_seed_input",
                )

            st.write("")
            run_restore_btn = st.button(
                "🚀 Bắt đầu phục hồi",
                type="primary",
                use_container_width=True,
                key="ddrm_run_restore_btn",
            )

        # ── KẾT QUẢ PHỤC HỒI ─────────────────────────────────────────────────
        if run_restore_btn:
            if input_pil is None:
                st.error("⚠️ Vui lòng chọn hoặc tải lên một bức ảnh trước khi bấm chạy!")
            else:
                deg = selected_task["deg"]
                sigma_0 = selected_task["sigma_0"]
                eta = selected_task["eta"]
                etaB = selected_task["etaB"]

                image_size = config.data.image_size
                device = torch.device(device_choice)

                torch.manual_seed(int(seed_val))
                np.random.seed(int(seed_val))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed_val))

                args = type("Args", (), {})()
                args.timesteps = int(timesteps)
                args.eta = float(eta)
                args.etaB = float(etaB)
                args.deg = deg
                if deg.startswith("sr"):
                    ratio = int(deg.replace("sr", ""))
                    args.sigma_0 = (float(sigma_0) * 2.0) / ratio
                elif deg.startswith("deblur") and not is_simulate:
                    args.sigma_0 = max(float(sigma_0), 0.01) * 2.0
                else:
                    args.sigma_0 = float(sigma_0) * 2.0

                diffusion = Diffusion(args, config, device=device)

                with st.spinner("⏳ Đang tính toán toán tử nghịch đảo và phục hồi ảnh bằng DDRM..."):
                    try:
                        input_tensor = transforms.ToTensor()(input_pil).unsqueeze(0).to(device)
                        inp_mask = None
                        if deg == "inp" and not is_simulate:
                            inp_mask = infer_inp_mask_from_image(input_tensor)

                        H_funcs = build_degradation(deg, config, device, inp_mask=inp_mask)

                        y_0, degraded_pil = prepare_measurement(
                            input_tensor=input_tensor,
                            config=config,
                            deg=deg,
                            device=device,
                            H_funcs=H_funcs,
                            sigma_0=float(sigma_0) * 2.0,
                            simulate=is_simulate,
                        )

                        x_init = torch.randn(
                            1,
                            config.data.channels,
                            config.data.image_size,
                            config.data.image_size,
                            device=device,
                        )

                        with torch.inference_mode():
                            restored = diffusion.sample_image(
                                x_init,
                                model,
                                H_funcs,
                                y_0,
                                args.sigma_0,
                                last=True,
                            )

                        restored_tensor = inverse_data_transform(config, restored).clamp(0, 1).squeeze(0).cpu()
                        restored_pil = transforms.ToPILImage()(restored_tensor)

                        st.success("🎉 Đã phục hồi ảnh thành công!")

                        st.markdown("### 📊 Kết quả so sánh")
                        comp_w = image_size * zoom_factor

                        if is_simulate:
                            res_c1, res_c2, res_c3 = st.columns(3)
                            with res_c1:
                                st.markdown(
                                    """
                                    <div class="img-card">
                                        <div class="img-card-title">1. Ảnh gốc ban đầu</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.image(input_pil.resize((comp_w, comp_w), Image.NEAREST), width=comp_w)

                            with res_c2:
                                st.markdown(
                                    """
                                    <div class="img-card">
                                        <div class="img-card-title">2. Ảnh suy thoái (Mô phỏng)</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.image(degraded_pil.resize((comp_w, comp_w), Image.NEAREST), width=comp_w)

                            with res_c3:
                                st.markdown(
                                    """
                                    <div class="img-card">
                                        <div class="img-card-title">3. Ảnh đã khôi phục (DDRM)</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.image(restored_pil.resize((comp_w, comp_w), Image.NEAREST), width=comp_w)
                        else:
                            res_c1, res_c2 = st.columns(2)
                            with res_c1:
                                st.markdown(
                                    """
                                    <div class="img-card">
                                        <div class="img-card-title">1. Ảnh hư tổn đầu vào</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.image(degraded_pil.resize((comp_w, comp_w), Image.NEAREST), width=comp_w)

                            with res_c2:
                                st.markdown(
                                    """
                                    <div class="img-card">
                                        <div class="img-card-title">2. Ảnh đã khôi phục (DDRM)</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.image(restored_pil.resize((comp_w, comp_w), Image.NEAREST), width=comp_w)

                        st.markdown("---")
                        st.markdown("#### ↔️ Kéo thanh trượt để so sánh trực tiếp")

                        slider_col_l, slider_col_c, slider_col_r = st.columns([1, 4, 1])
                        with slider_col_c:
                            compare_w = max(comp_w, 460)
                            render_compare_slider(
                                degraded_pil.resize((compare_w, compare_w), Image.NEAREST),
                                restored_pil.resize((compare_w, compare_w), Image.NEAREST),
                                width=compare_w,
                                height=compare_w,
                                key=f"comp_{deg}_{seed_val}",
                                label_left="Ảnh suy thoái",
                                label_right="DDRM Phục hồi",
                            )

                            buf = io.BytesIO()
                            restored_pil.save(buf, format="PNG")
                            st.download_button(
                                label="💾 Tải ảnh phục hồi (PNG)",
                                data=buf.getvalue(),
                                file_name=f"restored_{deg}_{seed_val}.png",
                                mime="image/png",
                                use_container_width=True,
                                key="ddrm_download_restored_btn",
                            )

                    except Exception as err:
                        st.error(f"Đã xảy ra lỗi trong quá trình phục hồi: {err}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: SINH ẢNH (DDIM & DDPM)
    # ═════════════════════════════════════════════════════════════════════════
    with tab_generate:
        gen_c1, gen_c2 = st.columns([1, 1], gap="large")

        with gen_c1:
            st.markdown("#### 1. Lựa chọn thuật toán & Số lượng")
            algo_choice = st.radio(
                "Phương pháp chạy:",
                [
                    "⚖️ So sánh cả hai (DDIM vs DDPM)",
                    "⚡ Chỉ chạy DDIM",
                    "🐢 Chỉ chạy DDPM",
                ],
                index=0,
                key="gen_algo_choice_radio",
            )

            is_compare = "So sánh" in algo_choice
            is_only_ddpm = "Chỉ chạy DDPM" in algo_choice

            total_samples = st.select_slider(
                "Số lượng ảnh:",
                options=[1, 2, 4],
                value=1 if (is_compare or is_only_ddpm) else 4,
                key="gen_total_samples_slider",
            )

        with gen_c2:
            st.markdown("#### 2. Điều chỉnh số bước & Seed")
            if is_compare:
                step_c1, step_c2 = st.columns(2)
                with step_c1:
                    ddim_steps = st.slider(
                        "Số bước DDIM:",
                        min_value=10,
                        max_value=100,
                        value=20,
                        step=5,
                        key="cmp_ddim_steps_slider",
                    )
                with step_c2:
                    ddpm_steps = st.select_slider(
                        "Số bước DDPM:",
                        options=[10, 20, 50, 100, 200, 500, 1000],
                        value=20,
                        key="cmp_ddpm_steps_slider",
                    )
            elif is_only_ddpm:
                ddpm_steps = st.select_slider(
                    "Số bước khử nhiễu DDPM (Timesteps):",
                    options=[10, 20, 50, 100, 200, 500, 1000],
                    value=1000,
                    key="single_ddpm_steps_slider",
                )
            else:
                ddim_steps = st.slider(
                    "Số bước khử nhiễu DDIM (Timesteps):",
                    min_value=10,
                    max_value=100,
                    value=20,
                    step=5,
                    key="single_ddim_steps_slider",
                )

            col_seed_num, col_seed_btn = st.columns([3, 2])
            with col_seed_btn:
                st.write("")
                st.write("")
                if st.button("🎲 Đổi Seed", key="gen_rand_seed_btn"):
                    st.session_state["gen_seed_input"] = int(np.random.randint(0, 999999))
            with col_seed_num:
                gen_seed = st.number_input(
                    "Mã số Seed:",
                    min_value=0,
                    max_value=999999,
                    value=1235,
                    step=1,
                    key="gen_seed_input",
                )

            st.write("")
            start_gen_btn = st.button(
                "🚀 Bắt đầu chạy",
                type="primary",
                use_container_width=True,
                key="gen_start_btn",
            )

        # ─────────────────────────────────────────────────────────────────────
        # XỬ LÝ KHI BẤM NÚT CHẠY
        # ─────────────────────────────────────────────────────────────────────
        if start_gen_btn:
            device = torch.device(device_choice)

            if is_compare:
                prog_c1, prog_c2 = st.columns(2, gap="large")
                with prog_c1:
                    st.markdown(f"**⚡ DDIM ({ddim_steps} bước)**")
                    ddim_pbar = st.progress(0.0)
                    ddim_status = st.empty()
                    ddim_preview = st.empty()
                with prog_c2:
                    st.markdown(f"**🐢 DDPM ({ddpm_steps} bước)**")
                    ddpm_pbar = st.progress(0.0)
                    ddpm_status = st.empty()
                    ddpm_preview = st.empty()

                try:
                    ddim_imgs, ddim_hist, ddim_time = sample_ddim(
                        model=model,
                        config=config,
                        timesteps=ddim_steps,
                        eta=0.0,
                        total_samples=total_samples,
                        seed=gen_seed,
                        device=device,
                        progress_bar=ddim_pbar,
                        status_text=ddim_status,
                        live_preview=ddim_preview,
                        method="ddim",
                    )
                    ddim_preview.empty()
                    ddim_status.success(f"✅ DDIM: {ddim_steps} bước ({ddim_time:.2f}s)")

                    ddpm_imgs, ddpm_hist, ddpm_time = sample_ddim(
                        model=model,
                        config=config,
                        timesteps=ddpm_steps,
                        eta=1.0,
                        total_samples=total_samples,
                        seed=gen_seed,
                        device=device,
                        progress_bar=ddpm_pbar,
                        status_text=ddpm_status,
                        live_preview=ddpm_preview,
                        method="ddpm",
                    )
                    ddpm_preview.empty()
                    ddpm_status.success(f"✅ DDPM: {ddpm_steps} bước ({ddpm_time:.2f}s)")

                    st.session_state["gen_last_mode"] = "compare"
                    st.session_state["cmp_results"] = {
                        "seed": gen_seed,
                        "samples": total_samples,
                        "ddim_steps": ddim_steps,
                        "ddpm_steps": ddpm_steps,
                        "ddim_imgs": ddim_imgs,
                        "ddpm_imgs": ddpm_imgs,
                        "ddim_hist": ddim_hist,
                        "ddpm_hist": ddpm_hist,
                        "ddim_time": ddim_time,
                        "ddpm_time": ddpm_time,
                    }
                except Exception as cmp_err:
                    st.error(f"Đã xảy ra lỗi: {cmp_err}")

            else:
                method_tag = "ddpm" if is_only_ddpm else "ddim"
                run_steps = ddpm_steps if is_only_ddpm else ddim_steps
                eta_val = 1.0 if is_only_ddpm else 0.0

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                live_c1, _ = st.columns([1, 2])
                with live_c1:
                    live_preview = st.empty()

                try:
                    with st.spinner(f"Đang chạy {method_tag.upper()} ({run_steps} bước)..."):
                        generated_images, step_history, elapsed_sec = sample_ddim(
                            model=model,
                            config=config,
                            timesteps=run_steps,
                            eta=eta_val,
                            total_samples=total_samples,
                            seed=gen_seed,
                            device=device,
                            progress_bar=progress_bar,
                            status_text=status_text,
                            live_preview=live_preview,
                            method=method_tag,
                        )

                    live_preview.empty()
                    status_text.success(f"✅ {method_tag.upper()}: Hoàn tất {run_steps} bước trong {elapsed_sec:.2f}s")

                    st.session_state["gen_last_mode"] = "single"
                    st.session_state["single_results"] = {
                        "method": method_tag.upper(),
                        "steps": run_steps,
                        "seed": gen_seed,
                        "imgs": generated_images,
                        "hist": step_history,
                        "time": elapsed_sec,
                    }
                except Exception as gen_err:
                    st.error(f"Đã xảy ra lỗi: {gen_err}")

        # ─────────────────────────────────────────────────────────────────────
        # HIỂN THỊ KẾT QUẢ
        # ─────────────────────────────────────────────────────────────────────
        image_size = config.data.image_size
        display_w = image_size * zoom_factor

        if is_compare and "cmp_results" in st.session_state:
            res = st.session_state["cmp_results"]
            r_seed = res["seed"]
            r_ddim_steps = res["ddim_steps"]
            r_ddpm_steps = res["ddpm_steps"]
            r_ddim_time = res["ddim_time"]
            r_ddpm_time = res["ddpm_time"]
            r_ddim_imgs = res["ddim_imgs"]
            r_ddpm_imgs = res["ddpm_imgs"]
            r_ddim_hist = res["ddim_hist"]
            r_ddpm_hist = res["ddpm_hist"]

            st.markdown("---")
            st.markdown("### 🖼️ Kết quả so sánh DDIM & DDPM")

            if r_ddpm_steps < 1000:
                ddpm_border_color = "#FECACA"
                ddpm_bg_color = "#FEF2F2"
                ddpm_title_color = "#DC2626"
                ddpm_accent = "#DC2626"
            else:
                ddpm_border_color = "#FDE68A"
                ddpm_bg_color = "#FFFBEB"
                ddpm_title_color = "#D97706"
                ddpm_accent = "#D97706"

            side_c1, side_c2 = st.columns(2, gap="large")

            with side_c1:
                st.markdown(
                    f"""
                    <div style="background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 12px; padding: 10px 16px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 15px; font-weight: 800; color: #3730A3;">⚡ DDIM ({r_ddim_steps} bước)</span>
                        <span style="background: #4F46E5; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 700;">⏱️ {r_ddim_time:.2f}s</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_ddim_step_scrubber(
                    step_history=r_ddim_hist,
                    display_w=display_w,
                    key=f"cmp_ddim_{r_seed}_{r_ddim_steps}",
                    label_prefix="⏩ Tiến trình:",
                    accent_color="#4F46E5",
                    badge_bg="#EEF2FF",
                    badge_text="#4338CA",
                    badge_border="#C7D2FE",
                )
                dl_ddim_cols = st.columns(len(r_ddim_imgs))
                for idx, pil_img in enumerate(r_ddim_imgs):
                    with dl_ddim_cols[idx]:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        st.download_button(
                            label=f"💾 Tải DDIM #{idx + 1}",
                            data=buf.getvalue(),
                            file_name=f"ddim_{r_ddim_steps}steps_seed{r_seed}_{idx + 1}.png",
                            mime="image/png",
                            key=f"dl_cmp_ddim_{idx}_{r_seed}_{r_ddim_steps}",
                            use_container_width=True,
                        )

            with side_c2:
                st.markdown(
                    f"""
                    <div style="background: {ddpm_bg_color}; border: 1px solid {ddpm_border_color}; border-radius: 12px; padding: 10px 16px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 15px; font-weight: 800; color: {ddpm_title_color};">🐢 DDPM ({r_ddpm_steps} bước)</span>
                        <span style="background: {ddpm_accent}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 700;">⏱️ {r_ddpm_time:.2f}s</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_ddim_step_scrubber(
                    step_history=r_ddpm_hist,
                    display_w=display_w,
                    key=f"cmp_ddpm_{r_seed}_{r_ddpm_steps}",
                    label_prefix="⏩ Tiến trình:",
                    accent_color=ddpm_accent,
                    badge_bg=ddpm_bg_color,
                    badge_text=ddpm_title_color,
                    badge_border=ddpm_border_color,
                )
                dl_ddpm_cols = st.columns(len(r_ddpm_imgs))
                for idx, pil_img in enumerate(r_ddpm_imgs):
                    with dl_ddpm_cols[idx]:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        st.download_button(
                            label=f"💾 Tải DDPM #{idx + 1}",
                            data=buf.getvalue(),
                            file_name=f"ddpm_{r_ddpm_steps}steps_seed{r_seed}_{idx + 1}.png",
                            mime="image/png",
                            key=f"dl_cmp_ddpm_{idx}_{r_seed}_{r_ddpm_steps}",
                            use_container_width=True,
                        )

            st.markdown("---")
            st.markdown("#### ↔️ Kéo thanh trượt để so sánh trực tiếp DDPM vs DDIM")
            sl_l, sl_c, sl_r = st.columns([1, 4, 1])
            with sl_c:
                compare_w = max(display_w, 420)
                render_compare_slider(
                    r_ddpm_imgs[0].resize((compare_w, compare_w), Image.NEAREST),
                    r_ddim_imgs[0].resize((compare_w, compare_w), Image.NEAREST),
                    width=compare_w,
                    height=compare_w,
                    key=f"split_cmp_{r_seed}_{r_ddim_steps}_{r_ddpm_steps}",
                    label_left=f"DDPM ({r_ddpm_steps} bước • {r_ddpm_time:.1f}s)",
                    label_right=f"DDIM ({r_ddim_steps} bước • {r_ddim_time:.2f}s)",
                )

        elif not is_compare and "single_results" in st.session_state:
            s_res = st.session_state["single_results"]
            used_method = s_res["method"]
            used_steps = s_res["steps"]
            used_seed = s_res["seed"]
            used_imgs = s_res["imgs"]
            used_hist = s_res["hist"]
            used_time = s_res["time"]

            st.markdown("---")
            st.markdown(f"### 🖼️ Kết quả {used_method} ({used_steps} bước • {used_time:.2f}s)")

            if used_hist:
                render_ddim_step_scrubber(
                    step_history=used_hist,
                    display_w=display_w,
                    key=f"single_scrub_{used_method}_{used_seed}_{used_steps}",
                    label_prefix="⏩ Tiến trình:",
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            dl_cols = st.columns(len(used_imgs))
            for idx, pil_img in enumerate(used_imgs):
                with dl_cols[idx]:
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    st.download_button(
                        label=f"💾 Tải ảnh #{idx + 1}",
                        data=buf.getvalue(),
                        file_name=f"{used_method.lower()}_{used_steps}steps_{used_seed}_{idx + 1}.png",
                        mime="image/png",
                        key=f"dl_final_{used_method}_{idx}_{used_seed}_{used_steps}",
                        use_container_width=True,
                    )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3: HƯỚNG DẪN & GIẢI THÍCH
    # ═════════════════════════════════════════════════════════════════════════
    with tab_guide:
        st.markdown(
            """
            ### 📚 So sánh nhanh DDRM, DDIM và DDPM

            | Tiêu chí | 🛠️ DDRM (Restoration) | ⚡ DDIM (Fast Generation) | 🐢 DDPM (Original Generation) |
            | :--- | :--- | :--- | :--- |
            | **Mục đích** | **Khôi phục ảnh** (sửa mờ, nhiễu, mất góc, tô màu) | **Sinh ảnh mới tốc độ cao** | **Sinh ảnh mới nguyên bản** (Ho et al., 2020) |
            | **Đầu vào** | Ảnh bị suy thoái $y_0$ | Nhiễu Gaussian $\\mathcal{N}(0, I)$ | Nhiễu Gaussian $\\mathcal{N}(0, I)$ |
            | **Cơ chế khử nhiễu** | Nghịch đảo toán tử suy thoái bằng SVD trong không gian Diffusion | Quá trình **phi Markov** (Non-Markovian), nhảy bước trực tiếp | Chuỗi **Markov bước đơn** cố định theo hệ số $\\beta_t$ gốc |
            | **Số bước cần thiết** | **20 bước** | **20 bước** (~0.8 giây) | **1000 bước** (~30 giây; nếu chạy 20 bước ảnh sẽ còn nhiễu) |
            | **Đầu ra** | Ảnh gốc được tái tạo sắc nét | Khuôn mặt AI mới hoàn chỉnh | Khuôn mặt AI mới (khi chạy đủ $T=1000$) |
            """
        )


if __name__ == "__main__":
    main()
