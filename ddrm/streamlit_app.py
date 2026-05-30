import os
import io
import base64
import yaml
import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torchvision.transforms as transforms

from models.diffusion import Model
from runners.diffusion import Diffusion
from datasets import data_transform, inverse_data_transform
from functions.svd_replacement import Denoising, Deblurring, Deblurring2D, Colorization, Inpainting


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def infer_inp_mask_from_image(input_tensor, threshold=1.0 / 255.0):
    gray = input_tensor.mean(dim=1).squeeze(0)
    return (gray > threshold).to(torch.uint8)


def build_degradation(deg, config, device, inp_mask=None):
    if deg == "deno":
        sigma_0 = 0.05
        return Denoising(config.data.channels, config.data.image_size, device)
    if deg.startswith("inp"):
        if deg == "inp_lolcat":
            root_dir = os.path.dirname(os.path.abspath(__file__))
            mask_path = os.path.join(root_dir, "inp_masks", "lolcat_extra.npy")
            mask = _load_inp_mask(mask_path, config.data.image_size, device)
        elif deg == "inp_lorem":
            root_dir = os.path.dirname(os.path.abspath(__file__))
            mask_path = os.path.join(root_dir, "inp_masks", "lorem3.npy")
            mask = _load_inp_mask(mask_path, config.data.image_size, device)
        elif inp_mask is not None:
            mask = inp_mask.to(device)
        else:
            mask = None

        if mask is None:
            missing_r = torch.randperm(config.data.image_size ** 2, device=device)
            missing_r = missing_r[: config.data.image_size ** 2 // 2].long() * config.data.channels
            missing = [missing_r + ch for ch in range(config.data.channels)]
            missing = torch.cat(missing, dim=0)
        else:
            missing = _mask_to_missing_indices(mask, config.data.channels, device)

        return Inpainting(config.data.channels, config.data.image_size, missing, device)
    if deg == "deblur_uni":
        kernel = torch.tensor([1 / 9] * 9, device=device)
        return Deblurring(kernel, config.data.channels, config.data.image_size, device)
    if deg == "deblur_gauss":
        sigma = 10
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel = torch.stack([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).flatten()
        return Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size, device)
    if deg == "deblur_aniso":
        sigma = 20
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel2 = torch.stack([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).flatten()
        sigma = 1
        pdf = lambda x: torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2], device=device))
        kernel1 = torch.stack([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).flatten()
        return Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, config.data.image_size, device)
    if deg == "color":
        return Colorization(config.data.image_size, device)
    raise ValueError(f"Unsupported degradation: {deg}")


def normalize_uploaded_image(image, image_size, deg):
    if deg == "color":
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return image, tensor


def prepare_measurement(input_tensor, config, deg, device, H_funcs=None):
    meas = data_transform(config, input_tensor.to(device))
    if deg.startswith("inp"):
        if H_funcs is None:
            raise ValueError("H_funcs is required for inpainting measurement.")
        y_0 = H_funcs.H(meas)
    elif deg == "color":
        if meas.shape[1] != 1:
            meas = meas.mean(dim=1, keepdim=True)
        y_0 = meas.reshape(meas.shape[0], -1)
    else:
        y_0 = meas.reshape(meas.shape[0], -1)
    return y_0, meas


def load_state_dict(model, ckpt_path, device):
    states = torch.load(ckpt_path, map_location=device)
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


def _pil_to_base64(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


def render_compare_slider(left_img, right_img, width, height, key):
        left_b64 = _pil_to_base64(left_img)
        right_b64 = _pil_to_base64(right_img)
        html = f"""
<style>
#comp-{key} {{
    position: relative;
    height: {height}px;
    width: {width}px;
    margin: 0;
}}
#comp-{key} .img-comp-img {{
    position: absolute;
    width: {width}px;
    height: {height}px;
    overflow: hidden;
}}
#comp-{key} img {{
    display: block;
    width: {width}px;
    height: {height}px;
    image-rendering: pixelated;
}}
#comp-{key} .img-comp-slider {{
    position: absolute;
    z-index: 10;
    cursor: ew-resize;
    width: 4px;
    background: #ffffff;
    border: 1px solid #111111;
    height: {height}px;
    top: 0;
    left: calc(50% - 2px);
}}
#comp-{key} .img-comp-handle {{
    position: absolute;
    top: calc(50% - 12px);
    left: -10px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #ffffff;
    border: 2px solid #111111;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
}}
</style>
<div id="comp-{key}">
    <div class="img-comp-img">
        <img src="data:image/png;base64,{left_b64}" />
    </div>
    <div class="img-comp-img img-comp-overlay" id="overlay-{key}">
        <img src="data:image/png;base64,{right_b64}" />
    </div>
    <div class="img-comp-slider" id="slider-{key}">
        <div class="img-comp-handle"></div>
    </div>
</div>
<script>
(function() {{
    var container = document.getElementById("comp-{key}");
    var overlay = document.getElementById("overlay-{key}");
    var slider = document.getElementById("slider-{key}");
    var width = {width};
    var clicked = false;

    function setPos(x) {{
        x = Math.max(0, Math.min(x, width));
        overlay.style.width = x + "px";
        slider.style.left = (x - 2) + "px";
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
        var pos = getCursorPos(e);
        setPos(pos);
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
        window.removeEventListener("touchmove", slideMove);
    }}

    overlay.style.width = (width / 2) + "px";
    slider.addEventListener("mousedown", slideReady);
    slider.addEventListener("touchstart", slideReady);
    window.addEventListener("mouseup", slideFinish);
    window.addEventListener("touchend", slideFinish);
}})();
</script>
"""
        components.html(html, height=height + 12, width=width)


@st.cache_resource(show_spinner=False)
def load_model_and_config(config_path, ckpt_path, device_type):
    config = load_config(config_path)
    device = torch.device(device_type)
    model = Model(config)
    load_state_dict(model, ckpt_path, device)
    model.to(device)
    model.eval()
    return config, model


def main():
    st.set_page_config(page_title="DDRM Streamlit", layout="wide")
    st.title("DDRM Image Restoration")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.abspath(os.path.join(root_dir, "..", "ckpt_708000.pth"))
    default_config = os.path.abspath(os.path.join(root_dir, "configs", "custom.yml"))

    with st.sidebar:
        st.header("Settings")
        config_path = st.text_input("Config path", value=default_config)
        ckpt_path = st.text_input("Checkpoint path", value=default_ckpt)
        deg_options = {
            "deno": "deno",
            "inp": "inp",
            "inp lolcat": "inp_lolcat",
            "inp lorem": "inp_lorem",
            "color": "color",
            "deblur_gauss": "deblur_gauss",
            "deblur uniform": "deblur_uni",
            "deblur anisotropic": "deblur_aniso",
        }
        deg_notes = {
            "deno": "Dùng khi ảnh bị nhiễu (noise) hạt, không bị mờ do blur.",
            "inp": "Dùng khi ảnh bị mất vùng (đen/trắng) không có thông tin, muốn tự suy mask từ ảnh.",
            "inp lolcat": "Dùng khi ảnh bị mất vùng theo mask lolcat có sẵn trong dự án.",
            "inp lorem": "Dùng khi ảnh bị mất vùng theo mask lorem có sẵn trong dự án.",
            "color": "Dùng khi ảnh xám cần tô màu (grayscale → color).",
            "deblur_gauss": "Dùng khi ảnh bị mờ do Gaussian blur.",
            "deblur uniform": "Dùng khi ảnh bị mờ do blur đều (kernel đều).",
            "deblur anisotropic": "Dùng khi ảnh bị mờ theo hướng (blur không đẳng hướng).",
        }
        deg_label = st.selectbox("Degradation", list(deg_options.keys()), index=0)
        deg = deg_options[deg_label]
        st.caption(deg_notes.get(deg_label, ""))
        timesteps = st.slider("Timesteps", min_value=10, max_value=1000, value=20, step=10)
        sigma_0 = 0.05 if deg == "deno" else 0.0
        eta = 0.85
        etaB = 1.0
        zoom_label = st.selectbox("Phóng to (nearest)", ["Tắt", "4x", "8x"], index=0)
        zoom_factor = 1 if zoom_label == "Tắt" else int(zoom_label.replace("x", ""))
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=1234, step=1)

    uploaded = st.file_uploader("Upload noisy image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

    if uploaded is None:
        st.info("Please upload an image to start.")
        return

    if not os.path.isfile(config_path):
        st.error("Config file not found.")
        return

    if not os.path.isfile(ckpt_path):
        st.error("Checkpoint file not found.")
        return

    device = get_device()
    st.caption(f"Device: {device}")

    try:
        config, model = load_model_and_config(config_path, ckpt_path, str(device))
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    image_size = config.data.image_size
    input_image, input_tensor = normalize_uploaded_image(Image.open(uploaded), image_size, deg)
    inp_mask = None
    if deg == "inp":
        inp_mask = infer_inp_mask_from_image(input_tensor)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        display_input = input_image
        if zoom_factor > 1:
            display_input = input_image.resize(
                (image_size * zoom_factor, image_size * zoom_factor),
                resample=Image.NEAREST,
            )
        st.image(display_input, width=image_size * zoom_factor, output_format="PNG")

    if st.button("Restore", type="primary"):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        args = type("Args", (), {})()
        args.timesteps = int(timesteps)
        args.eta = float(eta)
        args.etaB = float(etaB)
        args.deg = deg
        # Sigma_0 is expected in the same scale as y ([-1, 1]).
        args.sigma_0 = float(sigma_0) * 2.0

        diffusion = Diffusion(args, config, device=device)
        try:
            H_funcs = build_degradation(deg, config, device, inp_mask=inp_mask)
        except Exception as exc:
            st.error(f"Failed to build degradation: {exc}")
            return

        y_0, _ = prepare_measurement(input_tensor, config, deg, device, H_funcs=H_funcs)
        x_init = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=device,
        )

        with st.spinner("Restoring image..."):
            with torch.inference_mode():
                restored = diffusion.sample_image(
                    x_init,
                    model,
                    H_funcs,
                    y_0,
                    args.sigma_0,
                    last=True,
                )

        restored = inverse_data_transform(config, restored)
        restored_img = restored.detach().cpu().squeeze(0)
        restored_img = transforms.ToPILImage()(restored_img)

        with col2:
            st.subheader("Restored")
            display_restored = restored_img
            if zoom_factor > 1:
                display_restored = restored_img.resize(
                    (image_size * zoom_factor, image_size * zoom_factor),
                    resample=Image.NEAREST,
                )
            st.image(display_restored, width=image_size * zoom_factor, output_format="PNG")

        st.subheader("So sánh")
        compare_size = image_size * zoom_factor
        compare_input = input_image.resize(
            (compare_size, compare_size),
            resample=Image.NEAREST,
        )
        compare_restored = restored_img.resize(
            (compare_size, compare_size),
            resample=Image.NEAREST,
        )
        render_compare_slider(
            compare_input,
            compare_restored,
            compare_size,
            compare_size,
            key=f"compare-{deg}-{zoom_factor}",
        )


if __name__ == "__main__":
    main()
