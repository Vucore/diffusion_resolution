import os
import shutil
import numpy as np

# ==================== 1. CẤU HÌNH ĐƯỜNG DẪN ====================
source_dir = "/Users/nguyenngocminh/Documents/Vu/Ky8/TTNT_NC/Images"  # Folder chứa ảnh hiện tại của bạn
output_dir = "/Users/nguyenngocminh/Documents/Vu/Ky8/TTNT_NC/Test"                         # Folder Test sẽ được tạo ra để lưu ảnh Val

# Tùy chọn: True nếu muốn CẮT (Move), False nếu muốn SAO CHÉP (Copy)
MOVE_FILES = False 

# ==================== 2. TÁI TẠO LOGIC CHIA DỮ LIỆU ====================
# Tạo folder Test nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Lấy và sắp xếp danh sách file để đồng bộ thứ tự (giống PyTorch ImageFolder)
valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]
all_files.sort() 

num_items = len(all_files)
indices = list(range(num_items))

# Shuffle cố định với seed 42 y hệt lúc train
rng = np.random.RandomState(seed=42)
rng.shuffle(indices)

val_ratio = 0.05
split = int(num_items * (1.0 - val_ratio))
val_indices = indices[split:]

# Lọc ra danh sách tên file của tập Val
val_files = [all_files[i] for i in val_indices]

# ==================== 3. TIẾN HÀNH TRÍCH XUẤT VÀ LƯU ====================
print(f"Tìm thấy tổng cộng: {num_items} ảnh.")
print(f"Đang tiến hành xử lý {len(val_files)} ảnh thuộc tập Val...")

success_count = 0
for filename in val_files:
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(output_dir, filename)
    
    try:
        if MOVE_FILES:
            shutil.move(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path) # copy2 giữ nguyên metadata của ảnh
        success_count += 1
    except Exception as e:
        print(f"Lỗi khi xử lý file {filename}: {e}")

action_name = "Cắt (Move)" if MOVE_FILES else "Sao chép (Copy)"
print(f"\n[Hoàn thành] Đã {action_name} thành công {success_count}/{len(val_files)} ảnh vào folder '{output_dir}'.")