# ✨ AI Diffusion Studio — Khôi phục ảnh (DDRM) & Sinh ảnh (DDIM / DDPM)

Ứng dụng web tương tác xây dựng trên **Streamlit** và **PyTorch**, trình diễn hai bài toán cốt lõi của mô hình khuếch tán (Diffusion Models) trên tập ảnh khuôn mặt $64 \times 64$ (CelebA-HQ):

1. **Khôi phục ảnh không cần huấn luyện lại (DDRM — Denoising Diffusion Restoration Models)**: Giải các bài toán suy thoái tuyến tính bằng phân rã SVD.
2. **Sinh ảnh từ nhiễu & So sánh trực quan (DDIM vs DDPM)**: Đối chiếu trực tiếp tốc độ hội tụ giữa **DDIM** (~20 bước) và **DDPM** nguyên bản (cần $T = 1000$ bước).

---

## 🌟 Tính năng chính

### 🛠️ Tab 1: Phục hồi ảnh (DDRM)
- **9 tác vụ khôi phục ảnh**:
  - `sr4` — Tăng nét 4x (Super-Resolution x4)
  - `sr2` — Tăng nét 2x (Super-Resolution x2)
  - `deno` — Khử hạt nhiễu (Denoising)
  - `deblur_gauss` — Khử mờ ống kính (Gaussian Blur)
  - `deblur_uni` — Khử mờ đều (Uniform Blur)
  - `deblur_aniso` — Khử mờ rung tay (Motion Blur)
  - `inp` — Vá ảnh che khuyết (Inpainting)
  - `inp_lolcat` — Vá chữ nghệ thuật (Lolcat Mask)
  - `color` — Tô màu ảnh đen trắng (Colorization)
- **3 nguồn ảnh đầu vào**: Ảnh mẫu (`Test/`), Bộ ảnh hư tổn sẵn (`Test_Degraded/`), hoặc tải ảnh bất kỳ từ máy tính.
- **2 chế độ thực hiện**: Mô phỏng suy thoái từ ảnh gốc (hiển thị 3 cột) và Khôi phục trực tiếp ảnh hư tổn sẵn (hiển thị 2 cột).
- **Thanh trượt Before / After**: Kéo so sánh trực tiếp trước & sau khi khôi phục và tải xuống ảnh `.png`.

### 🎨 Tab 2: Sinh ảnh (DDIM & DDPM)
- **3 chế độ chạy**:
  - `⚖️ So sánh cả hai (DDIM vs DDPM)` trên cùng một `Seed` nhiễu khởi tạo $\mathcal{N}(0, I)$.
  - `⚡ Chỉ chạy DDIM` (lấy mẫu phi Markov — cho ảnh nét ở ~20 bước).
  - `🐢 Chỉ chạy DDPM` (chuỗi Markov nguyên bản Ho et al., 2020 — cần đủ $T = 1000$ bước mới khử sạch nhiễu).
- **Thanh trượt xem từng bước 60 FPS**: Kéo chuột để quan sát tức thì quá trình khử nhiễu từ Bước 0 (100% nhiễu) tới bước cuối.
- **Thanh trượt đặt chồng DDPM vs DDIM** khi chạy so sánh song song.

### 📖 Tab 3: Hướng dẫn & Giới thiệu
- Bảng tóm tắt so sánh nhanh giữa **DDRM**, **DDIM** và **DDPM**.

### ⚡ Tăng tốc phần cứng linh hoạt
- Tự động phát hiện và cho phép chuyển đổi giữa **Apple Silicon (`mps`)**, **NVIDIA GPU (`cuda`)**, và **`cpu`** ngay trên Sidebar.

---

## 📁 Cấu trúc thư mục dự án

```text
diffusion_resolution/
├── app.py                        # Điểm khởi chạy ứng dụng Streamlit
├── requirements.txt              # Danh sách thư viện Python cần thiết
├── .gitignore                    # Cấu hình loại trừ checkpoint lớn, .venv, cache
├── generate_degraded_dataset.py  # Script tạo bộ ảnh hư tổn mẫu Test_Degraded/
├── ckpt_708000.pth               # (Tự tải / đặt vào đây - KHÔNG đẩy lên Git)
├── ddrm/
│   ├── streamlit_app.py          # Giao diện & pipeline DDRM / DDIM / DDPM
│   ├── configs/custom.yml        # Cấu hình mô hình 64x64 (CelebA_HQ)
│   ├── functions/                # Toán tử suy thoái SVD & hàm lấy mẫu
│   ├── models/                   # Kiến trúc mạng UNet
│   ├── runners/                  # Bộ điều phối suy luận DDRM
│   └── README.md                 # Tài liệu gốc của DDRM CLI
├── ddim-custom/                  # Mã nguồn huấn luyện & lấy mẫu DDIM/DDPM CLI
│   └── README.md
├── Test/                         # Kho ảnh mẫu khuôn mặt 64x64
├── Test_Degraded/                # 90 ảnh hư tổn sẵn chia theo 9 thư mục lỗi
└── paper/                        # Các bài báo tham khảo (DDIM, DDRM, SR3, PGDiff)
```

---

## 💻 Yêu cầu hệ thống

- **Python**: `3.10` – `3.11` (khuyên dùng **Python 3.11**).
- **RAM**: Tối thiểu `8 GB`.
- **Phần cứng (khuyên dùng)**: Apple Silicon (`mps`) hoặc GPU NVIDIA (`cuda`) để chạy nhanh, đặc biệt khi demo DDPM ở $T = 1000$ bước (vẫn chạy được trên `cpu` nhưng sẽ chậm hơn).

---

## 🛠️ Hướng dẫn cài đặt từ đầu (sau khi `git clone`)

### Bước 1: Clone repository
```bash
git clone https://github.com/Vucore/diffusion_resolution.git
cd diffusion_resolution
```

### Bước 2: Tạo và kích hoạt môi trường ảo Python

- **Trên macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

- **Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Bước 3: Cài đặt thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý cho máy dùng NVIDIA GPU (Windows / Linux):** Nếu lệnh trên cài PyTorch bản CPU, hãy cài thêm PyTorch hỗ trợ CUDA từ [pytorch.org](https://pytorch.org/get-started/locally/) theo đúng phiên bản CUDA của máy trước hoặc sau khi chạy `requirements.txt`.

### Bước 4: Chuẩn bị tệp Checkpoint (`ckpt_708000.pth`)

> ⚠️ **Quan trọng:** File trọng số `ckpt_708000.pth` (~1.26 GB) bị loại khỏi Git thông qua `.gitignore` (vượt giới hạn 100 MB của GitHub). Người clone repo cần tự bổ sung file này trước khi chạy ứng dụng.

- **Tải checkpoint mẫu:** [Tải ckpt_708000.pth (Google Drive)](https://drive.google.com/file/d/1gq9pUu9SShxSr-htUHHx86MBz4fV8Tu1/view?usp=sharing)
- Sau khi tải về, đặt file vào **thư mục gốc dự án**:
  ```text
  diffusion_resolution/ckpt_708000.pth
  ```
- **Tuỳ chọn đường dẫn khác:** Nếu bạn lưu `.pth` ở thư mục khác (hoặc tự train mô hình bằng `ddim-custom/`), bạn vẫn có thể mở ứng dụng và đổi đường dẫn trong mục **`📁 Tệp Checkpoint & Config`** trên Sidebar bên trái.

---

## 🚀 Chạy ứng dụng Streamlit

Đảm bảo đã kích hoạt `.venv` và đang đứng tại thư mục gốc `diffusion_resolution/`, sau đó chạy:

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở tại địa chỉ mặc định: **http://localhost:8501**

**Mẹo sử dụng giao diện:**
- Ở Sidebar bên trái, bạn có thể đổi **Thiết bị tính toán** (`mps` / `cuda` / `cpu`) và **Kích thước hiển thị** (`2x`, `4x`, `6x`, `1x`).
- Nếu gặp lỗi tương thích phép toán trên macOS `mps`, hãy chuyển thiết bị về `cpu` ở Sidebar.
- Khi thử **DDPM 1000 bước** trên CPU, nên chọn **Số lượng ảnh = `1`** để giảm thời gian chờ.

---

## 🧪 Các script & CLI mở rộng (tùy chọn)

### 1. Tạo lại bộ ảnh hư tổn `Test_Degraded/`
Nếu muốn tái tạo 90 ảnh hư tổn mẫu từ `Test/` (10 ảnh × 9 loại suy thoái):
```bash
python generate_degraded_dataset.py
```

### 2. Huấn luyện hoặc sinh ảnh bằng CLI (`ddim-custom/`)
Xem hướng dẫn chi tiết tại [`ddim-custom/README.md`](ddim-custom/README.md). Ví dụ sinh ảnh nhanh bằng DDIM 20 bước:
```bash
cd ddim-custom
python main.py --config custom.yml --doc cmp_ddim --sample \
  --sample_type generalized --eta 0.0 --timesteps 20 --total_samples 1 \
  --ckpt ../ckpt_708000.pth -i ddim_compare --seed 1235
```

### 3. Chạy DDRM bằng CLI (`ddrm/`)
Xem hướng dẫn chi tiết về tham số dòng lệnh và cấu hình tại [`ddrm/README.md`](ddrm/README.md).

---

## 🔧 Xử lý sự cố thường gặp

| Vấn đề | Nguyên nhân & Cách khắc phục |
| :--- | :--- |
| **`❌ Không tìm thấy file model checkpoint`** | Chưa đặt `ckpt_708000.pth` ở thư mục gốc dự án hoặc sai đường dẫn. Hãy tải checkpoint vào đúng đường dẫn hoặc sửa đường dẫn trong **`📁 Tệp Checkpoint & Config`** trên Sidebar. |
| **Lỗi tương thích trên Mac (`mps`)** | Một số phép toán tensor chưa ổn định trên `mps`. Đổi **Thiết bị tính toán** sang `cpu` ở Sidebar bên trái. |
| **DDPM 1000 bước chạy chậm** | DDPM nguyên bản lặp 1000 bước UNet liên tiếp. Hãy bật `cuda`/`mps` nếu có GPU và chọn `Số lượng ảnh = 1` khi chạy 1000 bước. |

---

## 📚 Tài liệu tham khảo

Các bài báo gốc được lưu tham khảo trong thư mục `paper/`:
- **DDPM**: Ho et al. (2020), *Denoising Diffusion Probabilistic Models*
- **DDIM**: Song et al. (2021), *Denoising Diffusion Implicit Models*
- **DDRM**: Kawar et al. (2022), *Denoising Diffusion Restoration Models*
