"""
app.py - Streamlit demo for Off-Road Semantic Segmentation.
Run: py -m streamlit run app.py
"""

import os, sys, numpy as np
from PIL import Image
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import torch, torch.nn.functional as F
import config
from model import build_model
from utils import load_checkpoint, find_best_checkpoint, autocast_ctx
from augmentations import Resize, ToTensor, Normalize, Compose

st.set_page_config(page_title="Off-Road Segmentation", page_icon="🚗", layout="wide")

PALETTE = {
    "trees": (34,139,34), "lush_bushes": (0,200,0), "dry_grass": (210,180,80),
    "dry_bushes": (160,140,60), "ground_clutter": (120,80,40), "flowers": (255,105,180),
    "logs": (139,90,43), "rocks": (128,128,128), "landscape": (194,178,128), "sky": (135,206,235),
}
SAFE = {"trees","lush_bushes","dry_grass","landscape","sky","flowers"}

@st.cache_resource(show_spinner="Loading SegFormer model ...")
def load_model():
    model = build_model()
    ckpt = find_best_checkpoint()
    if ckpt: load_checkpoint(model, ckpt)
    model.eval()
    return model

def preprocess(image):
    t = Compose([Resize(config.IMAGE_SIZE), ToTensor(), Normalize()])
    img_t, _ = t(image.convert("RGB"), None)
    return img_t.unsqueeze(0)

@torch.no_grad()
def predict(model, image):
    inp = preprocess(image).to(config.DEVICE)
    with autocast_ctx(): logits = model(inp)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred, probs

def colorize(mask):
    h, w = mask.shape; rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, name in enumerate(config.CLASS_NAMES):
        rgb[mask == idx] = PALETTE.get(name, (255,255,255))
    return rgb

def safety_overlay(mask):
    h, w = mask.shape; rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, name in enumerate(config.CLASS_NAMES):
        rgb[mask == idx] = (0,200,0) if name in SAFE else (220,40,40)
    return rgb

def blend(base, overlay, alpha=0.45):
    base_r = np.array(Image.fromarray(base).resize((overlay.shape[1], overlay.shape[0])))
    return (base_r*(1-alpha) + overlay*alpha).clip(0,255).astype(np.uint8)

st.markdown("""<style>
.metric-card{background:#1e1e2e;border-radius:12px;padding:20px 24px;margin-bottom:12px;box-shadow:0 4px 20px rgba(0,0,0,.35)}
.metric-card h3{margin:0 0 6px;color:#cdd6f4;font-size:14px}.metric-card .value{font-size:32px;font-weight:700}
.good{color:#a6e3a1}.warn{color:#f9e2af}.bad{color:#f38ba8}
.pipeline-box{background:#1e1e2e;border-radius:12px;padding:18px 24px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.35);color:#cdd6f4;font-size:18px}
.analysis-card{background:#1e1e2e;border-radius:12px;padding:20px 24px;box-shadow:0 4px 20px rgba(0,0,0,.35);color:#cdd6f4;min-height:140px}
</style>""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:10px 0 20px"><h1 style="margin:0;font-size:2.4rem">🚗 Off-Road Semantic Segmentation System</h1><p style="color:#a6adc8;font-size:1.1rem;margin-top:4px">Transformer-based terrain understanding · SegFormer-B3 · 10 classes</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp"])
    overlay_alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    show_safety = st.checkbox("Show safe / obstacle map", value=True)
    st.markdown("---"); st.subheader("🎨 Class Legend")
    for name, colour in PALETTE.items():
        tag = "🟢" if name in SAFE else "🔴"
        hx = "#{:02x}{:02x}{:02x}".format(*colour)
        st.markdown(f'{tag} <span style="color:{hx};font-weight:600">{name}</span>', unsafe_allow_html=True)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    model = load_model()
    with st.spinner("🔍 Analyzing terrain ..."): mask, probs = predict(model, image)

    total = mask.size
    counts = {n: int(np.sum(mask==i)) for i,n in enumerate(config.CLASS_NAMES)}
    safe_pct = sum(v for k,v in counts.items() if k in SAFE) / total * 100

    seg_rgb = colorize(mask)
    img_np = np.array(image.resize((seg_rgb.shape[1], seg_rgb.shape[0])))
    overlay_np = blend(img_np, seg_rgb, overlay_alpha)

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown("**Input Image**"); st.image(img_np, use_container_width=True)
    with c2: st.markdown("**Segmentation**"); st.image(seg_rgb, use_container_width=True)
    with c3: st.markdown("**Overlay**"); st.image(overlay_np, use_container_width=True)

    if show_safety:
        st.markdown("**Safe / Obstacle Map**"); st.image(safety_overlay(mask), use_container_width=True)

    def _card(l,v,c): return f'<div class="metric-card"><h3>{l}</h3><div class="value {c}">{v}</div></div>'
    st.markdown("---")
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown(_card("Safe Terrain", f"{safe_pct:.1f}%", "good" if safe_pct>70 else ("warn" if safe_pct>40 else "bad")), unsafe_allow_html=True)
    with m2: st.markdown(_card("Model", "SegFormer-B3", "good"), unsafe_allow_html=True)
    with m3: st.markdown(_card("Classes", "10", "warn"), unsafe_allow_html=True)
    with m4:
        if safe_pct>70: v,c = "✅ Safe to Drive","good"
        elif safe_pct>40: v,c = "⚠️ Caution","warn"
        else: v,c = "🛑 Obstacle","bad"
        st.markdown(_card("Navigation", v, c), unsafe_allow_html=True)

    st.markdown("---"); st.subheader("📊 Per-Class Breakdown")
    for idx, name in enumerate(config.CLASS_NAMES):
        cols = st.columns([2,1,1,1])
        hx = "#{:02x}{:02x}{:02x}".format(*PALETTE.get(name,(200,200,200)))
        pct = counts[name]/total*100
        tag = "🟢 Safe" if name in SAFE else "🔴 Obstacle"
        cols[0].markdown(f'<span style="color:{hx};font-weight:600">■ {name}</span>', unsafe_allow_html=True)
        cols[1].write(f"{counts[name]:,}"); cols[2].write(f"{pct:.1f}%"); cols[3].write(tag)

    st.markdown("---")
    a1,a2 = st.columns(2)
    with a1: st.markdown('<div class="analysis-card"><h4 style="color:#a6e3a1">✅ Good Prediction</h4><p>Model performs well on <b>large terrain classes</b> — sky, trees, and landscape are segmented with high confidence.</p></div>', unsafe_allow_html=True)
    with a2: st.markdown('<div class="analysis-card"><h4 style="color:#f38ba8">❌ Failure Case</h4><p>Struggles with <b>small objects</b> (logs, rocks) and <b>extreme lighting</b> conditions (domain shift).</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="pipeline-box">📷 <b>Input</b> → 🔄 <b>Preprocess</b> → 🧠 <b>SegFormer-B3</b> → 🗺️ <b>Segmentation</b> → 🚗 <b>Navigation</b></div>', unsafe_allow_html=True)
else:
    st.info("👈 **Upload an image** in the sidebar to start terrain analysis.")
    st.markdown('<div class="pipeline-box">📷 <b>Input</b> → 🔄 <b>Preprocess</b> → 🧠 <b>SegFormer-B3</b> → 🗺️ <b>Segmentation</b> → 🚗 <b>Navigation</b></div>', unsafe_allow_html=True)
