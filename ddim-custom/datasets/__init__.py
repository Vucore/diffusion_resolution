"""
datasets/__init__.py
--------------------
Đây là module dataset đã được đơn giản hóa để hỗ trợ bộ dataset
tùy chỉnh: một folder chỉ chứa ảnh (không cần nhãn, không phân cấp).

Thay vì dùng CIFAR10 / CelebA / LSUN / FFHQ như bản gốc,
ta thêm class `CustomImageFolderDataset` đọc tất cả ảnh
từ một thư mục phẳng (flat folder).

Cấu hình trong file config yaml:
  data:
    dataset: "CUSTOM"
    data_dir: "path/to/your/images"   # thư mục chứa ảnh
    image_size: 64
    channels: 3
    random_flip: true
    rescaled: true
    uniform_dequantization: false
    gaussian_dequantization: false
    num_workers: 4
    val_ratio: 0.05   # tỷ lệ ảnh dùng làm validation (mặc định 5%)
"""

import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class CustomImageFolderDataset(Dataset):
    """
    Dataset đọc ảnh từ một thư mục phẳng (flat folder).
    Không cần nhãn, không cần cấu trúc con.
    Trả về tensor ảnh dạng float32 trong khoảng [0, 1].
    """

    def __init__(self, image_dir: str, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform

        # Thu thập tất cả file ảnh hợp lệ
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"Không tìm thấy ảnh nào trong thư mục: {image_dir}\n"
                f"Định dạng hỗ trợ: {IMAGE_EXTENSIONS}"
            )
        print(f"[Dataset] Đã tìm thấy {len(self.image_paths)} ảnh trong '{image_dir}'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # Trả về (ảnh, 0) để giữ tương thích với vòng lặp gốc "for x, y in loader"
        return img, 0


def get_dataset(args, config):
    """
    Hàm factory trả về (train_dataset, val_dataset).
    Hỗ trợ duy nhất: config.data.dataset == "CUSTOM"
    """
    if config.data.dataset != "CUSTOM":
        raise ValueError(
            f"Dataset '{config.data.dataset}' không được hỗ trợ trong phiên bản này.\n"
            "Hãy đặt config.data.dataset = 'CUSTOM' và config.data.data_dir = '<đường dẫn folder ảnh>'"
        )

    # ── Transforms ──────────────────────────────────────────────────────────
    base_transforms = [
        transforms.Resize(config.data.image_size),
        transforms.CenterCrop(config.data.image_size),
    ]
    if config.data.random_flip:
        base_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    base_transforms.append(transforms.ToTensor())  # → [0, 1]

    train_transform = transforms.Compose(base_transforms)
    val_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.CenterCrop(config.data.image_size),
        transforms.ToTensor(),
    ])

    # ── Tạo full dataset rồi split train/val ────────────────────────────────
    data_dir = config.data.data_dir
    full_dataset = CustomImageFolderDataset(data_dir, transform=train_transform)

    val_ratio = getattr(config.data, "val_ratio", 0.05)
    num_items = len(full_dataset)
    indices = list(range(num_items))

    # Shuffle cố định để reproducible
    rng = np.random.RandomState(seed=42)
    rng.shuffle(indices)

    split = int(num_items * (1.0 - val_ratio))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = Subset(full_dataset, train_indices)

    # Val dataset dùng val_transform (không flip)
    val_full_dataset = CustomImageFolderDataset(data_dir, transform=val_transform)
    val_dataset = Subset(val_full_dataset, val_indices)

    print(f"[Dataset] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset


# ── Data transforms (giữ nguyên logic gốc) ──────────────────────────────────

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    """Chuẩn hóa ảnh từ [0,1] → [-1,1] (hoặc logit transform)."""
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif getattr(config.data, "logit_transform", False):
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    """Giải chuẩn hóa ảnh từ [-1,1] → [0,1]."""
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if getattr(config.data, "logit_transform", False):
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
