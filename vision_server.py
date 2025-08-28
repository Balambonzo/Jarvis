#!/usr/bin/env python3
# vision_server.py
# Flask endpoint that runs YOLOv8 detection (Ultralytics) and BLIP captioning (HuggingFace)

import os
import io
import base64
import json
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Try to import heavy deps; if missing, the app will tell you what to install.
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
except Exception as e:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    torch = None

app = Flask(__name__)
CORS(app)

# Config
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")  # small net by default; change to yolov8s.pt / yolov8m.pt etc.
BLIP_MODEL = os.environ.get("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
DEVICE = os.environ.get("VISION_DEVICE", "cuda" if (torch is not None and torch.cuda.is_available()) else ("mps" if (torch is not None and torch.backends.mps.is_available()) else "cpu"))

# Lazy-load models
_yolo = None
_blip_processor = None
_blip_model = None

def ensure_yolo():
    global _yolo
    if _yolo is None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO module not available. Install via: pip install ultralytics")
        _yolo = YOLO(YOLO_MODEL)
    return _yolo

def ensure_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        if BlipProcessor is None or BlipForConditionalGeneration is None:
            raise RuntimeError("Transformers/BLIP not available. Install via: pip install transformers[torch] pillow")
        _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(DEVICE)
    return _blip_processor, _blip_model

def decode_data_url(data_url):
    # accepts either "data:image/jpeg;base64,..." or plain base64
    if data_url.startswith("data:"):
        header, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

@app.route("/api/vision", methods=["POST"])
def api_vision():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "missing image field"}), 400

        img_data = data["image"]
        # decode image
        try:
            pil = decode_data_url(img_data)
        except Exception as e:
            return jsonify({"error": f"failed_to_decode_image: {e}"}), 400

        # Run YOLO detection
        try:
            yolo = ensure_yolo()
            # Ultralytics accepts file path or numpy array; convert PIL to bytes
            # We pass PIL directly; YOLO will handle it
            results = yolo(pil, imgsz=640, conf=0.25)  # adjust conf
            # results is list of Results; we take first
            detections = []
            if len(results) > 0:
                r = results[0]
                # r.boxes is a Boxes object
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    # boxes.xyxy, boxes.conf, boxes.cls
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
                    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
                    names = r.names if hasattr(r, "names") else {}
                    for bb, conf, cls in zip(xyxy, confs, clss):
                        x1, y1, x2, y2 = [float(v) for v in bb]
                        label = names[int(cls)] if names and int(cls) in names else str(int(cls))
                        detections.append({"class": label, "conf": float(conf), "bbox": [x1, y1, x2, y2]})
        except Exception as e:
            # Non-fatal: continue with empty detections but report error in logs
            print("YOLO error:", e)
            detections = []

        # Run BLIP captioning (optional, if installed)
        caption = None
        try:
            proc, blip = ensure_blip()
            # prepare image
            inputs = proc(images=pil, return_tensors="pt").to(DEVICE)
            out = blip.generate(**inputs, max_new_tokens=40)
            caption = proc.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print("BLIP error (captioning):", e)
            caption = None

        return jsonify({
            "caption": caption,
            "detections": detections,
            "summary": caption or "no caption available"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "yolo_loaded": _yolo is not None, "blip_loaded": _blip_model is not None, "device": DEVICE})

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5001"))
    app.run(host=host, port=port, debug=True)
