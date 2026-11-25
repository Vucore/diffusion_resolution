"""
generate_degraded_dataset.py
----------------------------
Script tự động tạo bộ ảnh bị hư tổn (10 ảnh cho mỗi loại nhiễu/suy thoái)
từ tập ảnh gốc trong thư mục Test/.

Kết quả sẽ được lưu vào thư mục: Test_Degraded/<loại_nhiễu>/...
"""

import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Thiết lập đường dẫn
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DDRM_DIR = os.path.join(ROOT_DIR, "ddrm")
if DDRM_DIR not in sys.path:
    sys.path.insert(0, DDRM_DIR)

from datasets import data_transform, inverse_data_transform
from functions.svd_replacement import (
    Denoising,
    SuperResolution,
    Deblurring,
    Deblurring2D,
    Colorization,
    Inpainting,
)


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


def load_inp_mask(mask_path, image_size, device):
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


def mask_to_missing_indices(mask, channels, device):
    if mask.dim() == 2:
        mask = mask.reshape(-1)
    mask = mask.to(device)
    missing_r = torch.nonzero(mask == 0).long().reshape(-1) * channels
    missing = [missing_r + ch for ch in range(channels)]
    return torch.cat(missing, dim=0)


def build_operator(deg, config, device):
    channels = config.data.channels
    image_size = config.data.image_size

    if deg == "deno":
        return Denoising(channels, image_size, device)

    elif deg.startswith("sr"):
        ratio = int(deg.replace("sr", ""))
        return SuperResolution(channels, image_size, ratio, device)

    elif deg == "deblur_uni":
        kernel = torch.tensor([1.0 / 9.0] * 9, device=device)
        return Deblurring(kernel, channels, image_size, device)

    elif deg == "deblur_gauss":
        sigma = 10
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel = torch.stack([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).flatten()
        return Deblurring(kernel / kernel.sum(), channels, image_size, device)

    elif deg == "deblur_aniso":
        sigma = 20
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel2 = torch.stack([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).flatten()
        sigma = 1
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel1 = torch.stack([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).flatten()
        return Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), channels, image_size, device)

    elif deg == "color":
        return Colorization(image_size, device)

    elif deg.startswith("inp"):
        if deg == "inp_lolcat":
            mask_path = os.path.join(DDRM_DIR, "inp_masks", "lolcat_extra.npy")
            if os.path.exists(mask_path):
                mask = load_inp_mask(mask_path, image_size, device)
                missing = mask_to_missing_indices(mask, channels, device)
                return Inpainting(channels, image_size, missing, device)

        # Mặc định random mask 50% (luôn tạo trên CPU với seed cố định 1234 để đồng bộ tuyệt đối với streamlit_app)
        g = torch.Generator(device="cpu")
        g.manual_seed(1234)
        missing_r = torch.randperm(image_size ** 2, generator=g, device="cpu").to(device)
        missing_r = missing_r[: image_size ** 2 // 2].long() * channels
        missing = torch.cat([missing_r + ch for ch in range(channels)], dim=0)
        return Inpainting(channels, image_size, missing, device)

    raise ValueError(f"Không hỗ trợ deg: {deg}")


def create_degraded_image(clean_pil, deg, sigma_0, config, device, seed=1234):
    image_size = config.data.image_size
    img_rgb = clean_pil.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    input_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

    H_funcs = build_operator(deg, config, device)
    meas = data_transform(config, input_tensor)

    # Đặt seed cho nhiễu Gaussian sau khi khởi tạo H_funcs để không ảnh hưởng tới mask
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Đo lường suy thoái qua toán tử H
    y_0 = H_funcs.H(meas)
    if sigma_0 > 0:
        if deg.startswith("sr"):
            # Với Super-Resolution, thêm nhiễu ở mức pixel trung bình khối (không bị nhân lên r lần bởi H_pinv)
            ratio = int(deg.replace("sr", ""))
            y_0 = y_0 + (sigma_0 / ratio) * torch.randn_like(y_0)
        else:
            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)

    # 2. Khôi phục lại hình dạng biểu diễn ảnh suy thoái pinv_y_0 (kích thước 64x64)
    pinv_y_0 = H_funcs.H_pinv(y_0).view(1, config.data.channels, image_size, image_size)
    if deg.startswith("deblur"):
        pinv_y_0 = y_0.view(1, config.data.channels, image_size, image_size)
    elif deg.startswith("inp"):
        ones_meas = H_funcs.H(torch.ones_like(pinv_y_0))
        pinv_y_0 += H_funcs.H_pinv(ones_meas).reshape(*pinv_y_0.shape) - 1.0

    degraded_tensor = inverse_data_transform(config, pinv_y_0).clamp(0, 1).squeeze(0).cpu()
    return transforms.ToPILImage()(degraded_tensor)


def main():
    test_dir = os.path.join(ROOT_DIR, "Test")
    out_base_dir = os.path.join(ROOT_DIR, "Test_Degraded")
    config_path = os.path.join(DDRM_DIR, "configs", "custom.yml")

    if not os.path.isdir(test_dir):
        print(f"Lỗi: Không tìm thấy thư mục Test tại {test_dir}")
        return

    config = load_config(config_path)
    device = torch.device("cpu")  # Tạo ảnh dùng CPU để tương thích mọi máy

    # Lấy 10 ảnh đầu tiên từ Test/
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    all_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)])
    selected_files = all_files[:10]
    print(f"-> Đã chọn 10 ảnh mẫu từ tập Test: {selected_files}\n")

    # Danh mục 9 loại suy thoái (nhiễu)
    tasks = {
        "01_denoise_hat": {
            "deg": "deno",
            "name": "Nhiễu hạt Gaussian (Denoising)",
            "sigma_0": 0.05 * 2.0,  # Scale biên độ [-1, 1]
        },
        "02_sieu_phan_giai_sr2": {
            "deg": "sr2",
            "name": "Hạ độ phân giải 2x (Super-Resolution x2)",
            "sigma_0": 0.05 * 2.0,
        },
        "03_sieu_phan_giai_sr4": {
            "deg": "sr4",
            "name": "Hạ độ phân giải 4x (Super-Resolution x4)",
            "sigma_0": 0.05 * 2.0,
        },
        "04_mo_gaussian": {
            "deg": "deblur_gauss",
            "name": "Mờ ống kính (Gaussian Deblur)",
            "sigma_0": 0.0,
        },
        "05_mo_deu_uniform": {
            "deg": "deblur_uni",
            "name": "Mờ đều (Uniform Deblur)",
            "sigma_0": 0.0,
        },
        "06_mo_rung_tay_motion": {
            "deg": "deblur_aniso",
            "name": "Mờ chuyển động rung tay (Motion Deblur)",
            "sigma_0": 0.0,
        },
        "07_va_anh_inpainting": {
            "deg": "inp",
            "name": "Che khuyết ngẫu nhiên (Inpainting)",
            "sigma_0": 0.0,
        },
        "08_va_chu_lolcat": {
            "deg": "inp_lolcat",
            "name": "Che chữ nghệ thuật (Lolcat Mask)",
            "sigma_0": 0.0,
        },
        "09_den_trang_color": {
            "deg": "color",
            "name": "Ảnh đen trắng (Grayscale)",
            "sigma_0": 0.0,
        },
    }

    os.makedirs(out_base_dir, exist_ok=True)

    total_created = 0
    for folder_name, task_info in tasks.items():
        folder_path = os.path.join(out_base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        deg = task_info["deg"]
        sigma_0 = task_info["sigma_0"]
        name = task_info["name"]

        print(f"Đang tạo 10 ảnh cho loại: [{folder_name}] ({name})...")

        for idx, filename in enumerate(selected_files):
            clean_path = os.path.join(test_dir, filename)
            clean_pil = Image.open(clean_path)

            degraded_img = create_degraded_image(
                clean_pil=clean_pil,
                deg=deg,
                sigma_0=sigma_0,
                config=config,
                device=device,
                seed=1000 + idx,
            )

            # Lưu ảnh suy thoái với tên nguyên bản
            save_path = os.path.join(folder_path, filename)
            degraded_img.save(save_path, format="PNG")
            total_created += 1

        print(f"  ✓ Đã lưu xong 10 ảnh vào: {folder_path}")

    # Tạo file README trong Test_Degraded/ để hướng dẫn
    readme_path = os.path.join(out_base_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("BỘ DỮ LIỆU ẢNH HƯ TỔN (TEST DEGRADED DATASET)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Bộ ảnh này được tạo tự động từ 10 ảnh đầu tiên trong kho Test/,\n")
        f.write("mô phỏng chính xác 9 cơ chế suy thoái toán học của DDRM.\n\n")
        f.write("CÁCH SỬ DỤNG TRÊN ỨNG DỤNG STREAMLIT:\n")
        f.write("1. Chọn 'Ảnh đầu vào đã bị hư tổn sẵn (Khôi phục trực tiếp)'.\n")
        f.write("2. Tải lên một ảnh từ các folder dưới đây.\n")
        f.write("3. Chọn đúng tác vụ tương ứng trên giao diện để DDRM phục hồi:\n\n")
        for folder_name, task_info in tasks.items():
            f.write(f"  • Folder [{folder_name}] -> Chọn tác vụ: {task_info['name']}\n")

    print(f"\n🎉 HOÀN THÀNH TẤT CẢ! Đã tạo tổng cộng {total_created} ảnh suy thoái trong '{out_base_dir}'.")


if __name__ == "__main__":
    main()
