BỘ DỮ LIỆU ẢNH HƯ TỔN (TEST DEGRADED DATASET)
============================================================

Bộ ảnh này được tạo tự động từ 10 ảnh đầu tiên trong kho Test/,
mô phỏng chính xác 9 cơ chế suy thoái toán học của DDRM.

CÁCH SỬ DỤNG TRÊN ỨNG DỤNG STREAMLIT:
1. Chọn 'Ảnh đầu vào đã bị hư tổn sẵn (Khôi phục trực tiếp)'.
2. Tải lên một ảnh từ các folder dưới đây.
3. Chọn đúng tác vụ tương ứng trên giao diện để DDRM phục hồi:

  • Folder [01_denoise_hat] -> Chọn tác vụ: Nhiễu hạt Gaussian (Denoising)
  • Folder [02_sieu_phan_giai_sr2] -> Chọn tác vụ: Hạ độ phân giải 2x (Super-Resolution x2)
  • Folder [03_sieu_phan_giai_sr4] -> Chọn tác vụ: Hạ độ phân giải 4x (Super-Resolution x4)
  • Folder [04_mo_gaussian] -> Chọn tác vụ: Mờ ống kính (Gaussian Deblur)
  • Folder [05_mo_deu_uniform] -> Chọn tác vụ: Mờ đều (Uniform Deblur)
  • Folder [06_mo_rung_tay_motion] -> Chọn tác vụ: Mờ chuyển động rung tay (Motion Deblur)
  • Folder [07_va_anh_inpainting] -> Chọn tác vụ: Che khuyết ngẫu nhiên (Inpainting)
  • Folder [08_va_chu_lolcat] -> Chọn tác vụ: Che chữ nghệ thuật (Lolcat Mask)
  • Folder [09_den_trang_color] -> Chọn tác vụ: Ảnh đen trắng (Grayscale)
