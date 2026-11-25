import os
import sys

# Đảm bảo đường dẫn tới ddrm được thêm vào sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DDRM_DIR = os.path.join(ROOT_DIR, "ddrm")

if DDRM_DIR not in sys.path:
    sys.path.insert(0, DDRM_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from streamlit_app import main

if __name__ == "__main__":
    main()
