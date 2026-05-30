# DDIM cho Custom Dataset

Dự án này là phiên bản được tinh gọn từ **DDIM (Denoising Diffusion Implicit Models)** gốc, dành riêng cho việc huấn luyện (training) trên tập dữ liệu ảnh của riêng bạn một cách dễ dàng nhất.

Không cần định dạng phức tạp như TFRecords hay LMDB. Bạn chỉ cần gom tất cả hình ảnh vào **một thư mục duy nhất**.

---

## 🛠 Cài đặt

Yêu cầu máy tính có GPU (khuyên dùng NVIDIA) cài sẵn CUDA.

1. Khởi tạo môi trường Python (ảo):
   ```bash
   conda create -n ddim-custom python=3.10
   conda activate ddim-custom
   ```
2. Cài đặt PyTorch và các thư viện cần thiết:
   ```bash
   pip install torch torchvision
   pip install pyyaml tqdm tensorboard
   ```

---

## 🖼 Chuẩn bị Dữ liệu

Tạo một thư mục và thả toàn bộ hình ảnh bạn muốn huấn luyện vào đó (không cần chia thư mục con hay gán nhãn). Hình ảnh có thể là `.jpg`, `.png`, `.jpeg`, v.v.

Ví dụ cấu trúc thư mục:
```text
my_dataset/
├── img_001.jpg
├── img_002.jpg
├── whatever_name.png
└── ...
```

---

## 🚀 Hướng dẫn Sử dụng

### 1. Bắt đầu Huấn luyện (Training)

Chạy lệnh sau để bắt đầu train. Mã nguồn sẽ sử dụng cấu hình mặc định trong `configs/custom.yml`.

```bash
python main.py --config custom.yml --doc demo_run --data_dir path/to/my_dataset
```

**Giải thích tham số:**
- `--config custom.yml`: File cấu hình model, hyperparameter, batch_size... nằm trong thư mục `configs/`.
- `--doc demo_run`: Tên của experiment. Các checkpoint, log, tensorboard sẽ được lưu trong folder `exp/logs/demo_run`.
- `--data_dir`: Đường dẫn tuyệt đối hoặc tương đối tới thư mục chứa ảnh.

### 2. Tiếp tục Huấn luyện (Resume)

Nếu tiến trình bị dừng hoặc bạn muốn train thêm từ file `ckpt.pth` lưu tự động, hãy thêm tham số `--resume_training`:

```bash
python main.py --config custom.yml --doc demo_run --data_dir path/to/my_dataset --resume_training
```

### 3. Sinh ảnh (Sampling / Inference)

Sau khi mô hình đã hội tụ mượt mà (bạn có thể theo dõi qua TensorBoard), bạn bắt đầu dùng nó sinh ra ảnh mới (fake).

```bash
python main.py --config custom.yml --doc demo_run --sample --timesteps 100 --eta 0.0
```

**Giải thích tham số:**
- `--sample`: Kích hoạt chế độ sinh ảnh thay vì huấn luyện.
- `--timesteps 100`: Số bước đảo ngược quá trình nhiễu. Càng ít bước (vd: 50, 100) sinh càng nhanh nhưng chất lượng có thể thấp hơn. Tối đa là 1000.
- `--eta 0.0`: Sinh kiểu DDIM (deterministic). Nếu đặt `1.0` sẽ giống DDPM (stochastic).
- Ảnh sinh ra sẽ lưu ở `exp/image_samples/samples/`. Bạn có thể đổi tên thư mục này bằng `-i <tên folder>`.

---

## ⚙️ Tùy chỉnh Cấu hình (`configs/custom.yml`)

Bạn có thể chỉnh sửa `configs/custom.yml` cho phù hợp cấu hình máy:

- **`data.image_size`**: Kích thước ảnh resize để model học (mặc định đang để `64` hoặc `128`). Chú ý: tăng kích thước cần nhiều VRAM GPU hơn rất nhiều.
- **`training.batch_size`**: Số lượng ảnh đưa vào mỗi bước tối ưu. Giảm xuống nếu bạn bị lỗi **Out of Memory (OOM)**.
- **`sampling.total_samples`**: Đặt số lượng ảnh bạn muốn trả về khi chạy `--sample`.

---

## 📊 Theo dõi Huấn luyện

Mở TensorBoard để xem biểu đồ Loss (sai số):

```bash
tensorboard --logdir exp/tensorboard/
```
Sau đó truy cập http://localhost:6006 trên trình duyệt.
