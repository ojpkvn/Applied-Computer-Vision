# ===============================================================
# EcoSort | 04_app_gradio.py | Waste-sorting web app (L8)
# Tabs: 1) Upload (ResNet18 waste classifier OR YOLOv8n)  2) Webcam (YOLOv8n)
# Run:  python 04_app_gradio.py     (needs: pip install gradio ultralytics)
# ===============================================================
import csv
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from ultralytics import YOLO

# ---------------------------------------------------------------
# 1) Paths and settings
# ---------------------------------------------------------------
BASE_DIR  = Path("EcoSort_Files")
ART_DIR   = BASE_DIR / "artifacts"
SAVE_DIR  = BASE_DIR / "app_predictions"
LOG_CSV   = SAVE_DIR / "prediction_log.csv"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE         = torch.device("cpu")
IMG_SIZE       = 224
CONF_THRESHOLD = 0.60     # below this the app says "Unsure"

BIN_ADVICE = {   # edit to match your local recycling rules (and cite the source)
    "cardboard": "♻️ Recycling bin (flatten the box first)",
    "glass":     "♻️ Recycling bin (empty and rinse)",
    "metal":     "♻️ Recycling bin (empty and rinse)",
    "paper":     "♻️ Recycling bin (keep it dry and clean)",
    "plastic":   "♻️ Recycling bin (empty and rinse)",
    "trash":     "🗑️ General waste bin",
}

val_tfms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------
# 2) Load models
# ---------------------------------------------------------------
PREFERRED_CKPT = None   # e.g. "expD2_best.pth" to deploy the Experiment D model; None = best val accuracy of A/B/C

def pick_best_checkpoint():
    """Use PREFERRED_CKPT if set, else the checkpoint with the highest saved validation accuracy (A, B or C)."""
    if PREFERRED_CKPT and (ART_DIR / PREFERRED_CKPT).exists():
        print("✅ Using", PREFERRED_CKPT, "(PREFERRED_CKPT)"); return ART_DIR / PREFERRED_CKPT
    best, best_acc = None, -1.0
    for name in ["expA_best.pth", "expB_best.pth", "expC_best.pth"]:
        p = ART_DIR / name
        if p.exists():
            acc = torch.load(p, map_location=DEVICE).get("val_acc", 0.0)
            print(f"   found {name}: val_acc={acc:.3f}")
            if acc > best_acc:
                best, best_acc = p, acc
    if best is None:
        raise FileNotFoundError(f"No checkpoint in {ART_DIR}. Run the training parts (Part 2 and Part 4) of EcoSort_Full_Project.ipynb first.")
    print("✅ Using", best.name)
    return best

def load_resnet(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(ckpt["classes"]))
    net.load_state_dict(ckpt["state_dict"])
    return net.to(DEVICE).eval(), ckpt["classes"]

CKPT = pick_best_checkpoint()
resnet, CLASSES = load_resnet(CKPT)

# Optional: use the TorchScript model from notebook 02 (L7) for faster inference.
# Safety check: it is only used if its output matches the standard model, so a stale
# TorchScript file (traced from a different checkpoint) is never used by mistake.
USE_TORCHSCRIPT = True
TS_PATH = ART_DIR / "ecosort_resnet18_ts.pt"
if USE_TORCHSCRIPT and TS_PATH.exists():
    ts_model = torch.jit.load(str(TS_PATH), map_location=DEVICE).eval()
    probe = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        same = torch.allclose(resnet(probe), ts_model(probe), atol=1e-4)
    if same:
        resnet = ts_model
        print("✅ Using TorchScript model:", TS_PATH.name)
    else:
        print("⚠️ TorchScript file does not match the checkpoint, using the standard model")
else:
    print("ℹ️ Using the standard PyTorch model")

yolo = YOLO("yolov8n.pt")      # pre-trained COCO detector (L4)
print("✅ Models loaded | classes:", CLASSES)

# ---------------------------------------------------------------
# 3) Helper functions
# ---------------------------------------------------------------
@torch.no_grad()
def classify(pil_img, k=3):
    t0 = time.perf_counter()
    x = val_tfms(ImageOps.exif_transpose(pil_img).convert("RGB")).unsqueeze(0).to(DEVICE)  # fix phone-photo rotation
    probs = torch.softmax(resnet(x), dim=1)[0]
    top_p, top_i = torch.topk(probs, min(k, len(CLASSES)))
    ms = (time.perf_counter() - t0) * 1000
    return [(CLASSES[i], float(p)) for p, i in zip(top_p, top_i)], ms

def run_yolo(pil_img, conf=0.25):
    t0 = time.perf_counter()
    result = yolo.predict(np.array(pil_img.convert("RGB")), conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    annotated = result.plot()[:, :, ::-1]          # BGR -> RGB
    names = [result.names[int(c)] for c in result.boxes.cls]
    return annotated, names, ms

def log_prediction(filename, model_name, label, confidence, latency_ms):
    new_file = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "file", "model", "prediction", "confidence", "latency_ms"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), filename, model_name,
                    label, "" if confidence is None else f"{confidence:.3f}", f"{latency_ms:.1f}"])

# ---------------------------------------------------------------
# 4) Gradio callbacks
# ---------------------------------------------------------------
def upload_infer(img, model_choice):
    if img is None:
        return None, "⚠️ Please upload an image first."
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"upload_{stamp}.jpg"
    img.convert("RGB").save(SAVE_DIR / fname)

    if model_choice.startswith("ResNet"):
        top, ms = classify(img, k=3)
        label, conf = top[0]
        if conf < CONF_THRESHOLD:
            verdict = f"❓ Unsure (best guess: {label}, {conf:.0%}). Try a clearer photo."
        else:
            verdict = f"✅ {label.upper()} ({conf:.0%})\n➡️ {BIN_ADVICE.get(label, 'Check local rules')}"
        lines = "\n".join(f"   {i+1}. {c}: {p:.1%}" for i, (c, p) in enumerate(top))
        log_prediction(fname, "ResNet18", label, conf, ms)
        return img, f"{verdict}\n\nTop-3:\n{lines}\n\n⏱️ {ms:.1f} ms"

    annotated, names, ms = run_yolo(img)
    summary = ", ".join(sorted(set(names))) if names else "No objects detected"
    log_prediction(fname, "YOLOv8n", summary, None, ms)
    return annotated, f"🔍 Detected: {summary}\n⏱️ {ms:.1f} ms"

def webcam_infer(frame):
    if frame is None:
        return None
    annotated, _, _ = run_yolo(Image.fromarray(frame), conf=0.35)
    return annotated

# ---------------------------------------------------------------
# 5) Build the interface (L8 layout)
# ---------------------------------------------------------------
with gr.Blocks(title="EcoSort: Waste Sorting Assistant", theme=None) as demo:
    gr.Markdown("# ♻️ EcoSort: Waste Sorting Assistant\n"
                "**Upload** a photo of a waste item to find out which bin it belongs in, "
                "or use the **webcam** for live object detection.")
    with gr.Tabs():
        with gr.Tab("🖼️ Upload (Waste Classifier / YOLO)"):
            with gr.Row():
                inp = gr.Image(type="pil", label="Upload a waste item")
                model_choice = gr.Radio(
                    choices=["ResNet18 (Waste Classification)", "YOLOv8n (Object Detection)"],
                    value="ResNet18 (Waste Classification)", label="Choose Model")
            with gr.Row():
                out = gr.Image(label="Output", interactive=False)
                msg = gr.Textbox(label="Result / Bin Advice", lines=8, interactive=False)
            run_btn = gr.Button("▶️ Run Inference", variant="primary")
            run_btn.click(fn=upload_infer, inputs=[inp, model_choice], outputs=[out, msg])

        with gr.Tab("📸 Webcam (YOLO)"):
            gr.Markdown("Live **YOLOv8n** detection (pre-trained COCO classes).")
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
            cam_out = gr.Image(label="Detections", interactive=False)
            cam.stream(fn=webcam_infer, inputs=cam, outputs=cam_out, time_limit=None)

    gr.Markdown(f"---\n### ℹ️ Notes\n- ResNet18 fine-tuned on: {', '.join(CLASSES)}.\n"
                f"- Predictions below {CONF_THRESHOLD:.0%} confidence are marked 'Unsure'.\n"
                f"- Every upload is logged to `{LOG_CSV}`.")

demo.queue()
demo.launch()
