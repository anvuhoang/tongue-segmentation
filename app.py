from pathlib import Path
from io import BytesIO
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3


# ==================================================
# Configuration
# ==================================================
PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "checkpoints"

CHECKPOINTS = {
    "U-Net": CHECKPOINT_DIR / "best_unet.pth",
    "DeepLabV3": CHECKPOINT_DIR / "best_deeplabv3.pth",
}

IMAGE_SIZE = (288, 384)  # Height, Width
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="Tongue Segmentation",
    page_icon="🩺",
    layout="wide",
)

st.title("Tongue Image Segmentation")
st.write(
    "So sánh hai mô hình phân đoạn lưỡi: "
    "U-Net và DeepLabV3."
)

if DEVICE.type == "cuda":
    st.caption(f"Device: {torch.cuda.get_device_name(0)}")
else:
    st.caption("Device: CPU")


# ==================================================
# Load models
# ==================================================
@st.cache_resource
def load_model(model_name):
    checkpoint_path = CHECKPOINTS[model_name]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    if model_name == "U-Net":
        model = UNet()
    elif model_name == "DeepLabV3":
        model = DeepLabV3()
    else:
        raise ValueError(f"Model không được hỗ trợ: {model_name}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    epoch = checkpoint.get("epoch")
    return model, epoch


# ==================================================
# Preprocess
# ==================================================
def preprocess_image(image):
    original_image = image.convert("RGB")

    resized = original_image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized, dtype=np.float32
    ) / 255.0

    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1).copy()
    ).unsqueeze(0)

    return original_image, image_tensor.to(DEVICE)


# ==================================================
# Prediction
# ==================================================
def predict(model, image_tensor):
    # Warm-up is excluded from the measured inference time.
    with torch.inference_mode():
        _ = model(image_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = probabilities[0, 0] >= THRESHOLD

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    mask_array = mask.cpu().numpy().astype(np.uint8) * 255
    return mask_array, elapsed_ms


def create_overlay(original_image, mask_array):
    original_array = np.asarray(original_image).copy()
    overlay = original_array.copy()

    foreground = mask_array > 0
    red = np.array([255, 40, 40], dtype=np.float32)

    overlay[foreground] = (
        0.55 * original_array[foreground].astype(np.float32)
        + 0.45 * red
    ).astype(np.uint8)

    return overlay


def resize_mask(mask_array, original_size):
    mask = Image.fromarray(mask_array)
    mask = mask.resize(original_size, Image.Resampling.NEAREST)
    return np.asarray(mask)


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.header("Settings")

    selected_models = st.multiselect(
        "Select model(s)",
        options=["U-Net", "DeepLabV3"],
        default=["U-Net", "DeepLabV3"],
    )

    st.write(f"Input size: {IMAGE_SIZE[1]} × {IMAGE_SIZE[0]}")
    st.write(f"Threshold: {THRESHOLD}")
    st.write(f"Device: {DEVICE.type.upper()}")

    st.caption(
        "Inference time excludes model loading and image preprocessing."
    )


# ==================================================
# Upload
# ==================================================
uploaded_file = st.file_uploader(
    "Upload a tongue image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        original_size = image.size
        original_image, image_tensor = preprocess_image(image)

        st.subheader("Input image")
        st.image(original_image, use_container_width=True)
        st.caption(
            f"Original dimensions: {original_size[0]} × {original_size[1]} px"
        )

        if not selected_models:
            st.warning("Please select at least one model.")
            st.stop()

        if st.button("Run segmentation", type="primary"):
            results = {}

            progress = st.progress(0)

            for index, model_name in enumerate(selected_models):
                with st.spinner(f"Running {model_name}..."):
                    model, epoch = load_model(model_name)
                    mask_small, elapsed_ms = predict(
                        model, image_tensor
                    )
                    mask_original = resize_mask(
                        mask_small, original_size
                    )
                    overlay = create_overlay(
                        original_image, mask_original
                    )

                    results[model_name] = {
                        "mask": mask_original,
                        "overlay": overlay,
                        "time_ms": elapsed_ms,
                        "epoch": epoch,
                    }

                    # Free GPU memory between model runs.
                    del model
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

                progress.progress((index + 1) / len(selected_models))

            st.success("Segmentation completed.")

            columns = st.columns(len(results))

            for column, (model_name, result) in zip(
                columns, results.items()
            ):
                with column:
                    st.subheader(model_name)

                    if result["epoch"] is not None:
                        st.caption(
                            f"Best checkpoint epoch: {result['epoch']}"
                        )

                    st.image(
                        result["mask"],
                        caption="Predicted mask",
                        clamp=True,
                        use_container_width=True,
                    )

                    st.image(
                        result["overlay"],
                        caption="Overlay",
                        use_container_width=True,
                    )

                    st.metric(
                        "Inference time",
                        f'{result["time_ms"]:.2f} ms',
                    )

                    buffer = BytesIO()
                    Image.fromarray(result["mask"]).save(
                        buffer, format="PNG"
                    )

                    st.download_button(
                        label=f"Download {model_name} mask",
                        data=buffer.getvalue(),
                        file_name=(
                            f"{model_name.lower().replace('-', '')}"
                            "_predicted_mask.png"
                        ),
                        mime="image/png",
                        key=f"download_{model_name}",
                    )

            if len(results) == 2:
                st.subheader("Inference time comparison")

                unet_time = results["U-Net"]["time_ms"]
                deeplab_time = results["DeepLabV3"]["time_ms"]

                c1, c2 = st.columns(2)
                c1.metric("U-Net", f"{unet_time:.2f} ms")
                c2.metric("DeepLabV3", f"{deeplab_time:.2f} ms")

                st.caption(
                    "Thời gian có thể thay đổi theo phần cứng, "
                    "tình trạng GPU và lần chạy. Đây không phải "
                    "metric đánh giá độ chính xác phân đoạn."
                )

    except Exception as error:
        st.error(f"Không thể xử lý ảnh: {error}")
        st.exception(error)
