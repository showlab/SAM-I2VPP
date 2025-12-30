# -*- coding: utf-8 -*-
# @FileName: web_annotation_tool.py
# @Time    : 11/10/25 19:40
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import io
import base64
import logging
from typing import Dict, Tuple, List

from flask import Flask, request, jsonify, Response
import numpy as np
from PIL import Image
import cv2
import torch

from sam2.build_sam import build_sam2_video_predictor

# ==================== Configuration ====================
VIDEO_DIR = "data/JPEGImages_24fps/sav_017171"
MODEL_CFG = "configs/i2vpp-infer.yaml"
CHECKPOINT = "checkpoints/sam-i2vpp_32gpu.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNSAMPLE_STRIDE = 4
SAVE_DIR = VIDEO_DIR.replace("/JPEGImages_24fps/", "/Annotations_6fps/")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAM-I2V++_Annotation_Tool")

app = Flask(__name__)

frame_paths = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".jpg")])
if not frame_paths:
    raise RuntimeError(f"No .jpg frames found in {VIDEO_DIR}")
frame_cache: Dict[int, np.ndarray] = {}

# ==================== Keyframe Indices ====================
key_frame_indices = [i for i in range(len(frame_paths)) if i % DOWNSAMPLE_STRIDE == 0]
if not key_frame_indices:
    raise RuntimeError("No key frames detected!")

def load_frame(idx: int) -> np.ndarray:
    if idx in frame_cache:
        return frame_cache[idx]
    path = os.path.join(VIDEO_DIR, frame_paths[idx])
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame_cache[idx] = img
    return img

logger.info("Building SAM-I2V++ predictor (this may take a while)...")
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)
inference_state = predictor.init_state(video_path=VIDEO_DIR)
logger.info("Predictor ready.")

# The current frame_index refers to the keyframe index, not the index of all frames
frame_index: int = 0
prompt_cache: Dict[int, Dict[int, Tuple[List[List[int]], List[int]]]] = {}
mask_results: Dict[int, Dict[int, np.ndarray]] = {}

def is_key_frame(idx: int) -> bool:
    return (idx % DOWNSAMPLE_STRIDE) == 0

def _ensure_mask_2d_and_match(mask) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        if mask.shape[0] in (1,):
            mask = mask[0]
        elif mask.shape[2] in (1,):
            mask = mask[:, :, 0]
        else:
            mask = np.max(mask, axis=0)
    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask ndim after squeeze: {mask.shape}")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    return mask

def overlay_mask_on_image(img: np.ndarray, mask: np.ndarray, color=(0,255,0), alpha=0.5) -> np.ndarray:
    mask = _ensure_mask_2d_and_match(mask)
    h_img, w_img = img.shape[:2]
    if mask.shape != (h_img, w_img):
        mask = cv2.resize(mask.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = img.copy()
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[mask] = color
    out[mask] = cv2.addWeighted(img[mask], 1 - alpha, overlay[mask], alpha, 0)
    return out

def encode_image_to_data_url(img: np.ndarray, fmt="PNG") -> str:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return "data:image/{};base64,{}".format(fmt.lower(), base64.b64encode(buf.getvalue()).decode("utf-8"))

def current_display_image(idx: int) -> np.ndarray:
    img = load_frame(idx)
    if is_key_frame(idx) and idx in mask_results and mask_results[idx]:
        disp = img.copy()
        for _oid, m in mask_results[idx].items():
            try:
                disp = overlay_mask_on_image(disp, m, color=(0,255,0), alpha=0.5)
            except Exception as e:
                logger.exception(f"overlay failed for frame {idx} oid {_oid}: {e}")
        return disp
    return img

INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"/><title>SAM-I2V++ Annotation Tool</title>
<style>
body{font-family:Arial;padding:12px}
#img-container{display:flex;gap:20px}
.left-img-box{flex:3;display:flex;flex-direction:column;align-items:center;}
.right-img-box{flex:1;display:flex;flex-direction:column;align-items:center;}
img.frame-img-big{border:1px solid #1976d2;max-height:640px;width:auto;max-width:100%}
img.frame-img-small{border:1px solid #ccc;max-height:300px;width:auto;max-width:100%}
@media (max-width:900px){
    .left-img-box{flex:2;}
    .right-img-box{flex:1;}
}
input#goto_frame { width:70px; }
</style></head><body>
<h3>SAM-I2V++ Annotation Tool</h3>
<div style="display:flex;justify-content:space-between;align-items:center;gap:30px;">
  <div>
    <button id="prev">← Prev</button>
    <button id="next">Next →</button>
    <button id="save" style="background:#1976d2;color:#fff">Save</button>
  </div>
  <div>
    <button id="prop">Propagate</button>
    <button id="clear" style="background:#d32f2f;color:#fff">Clear</button>
  </div>
</div>
<div>
  <span id="fname"></span>
  <label style="margin-left:10px;">
    Jump to: <input id="goto_frame" type="number" min="0" style="width:60px"/>
    <button id="goto_btn">Go</button>
  </label>
</div>
<div id="img-container">
  <div class="left-img-box">
    <div style="text-align:center">Click on the object to annotate — Left-click for positive (+), Right-click for negative (−)</div>
    <img id="img" class="frame-img-big" src="" alt="overlay frame"/>
  </div>
</div>
<div>
Object ID: <input id="obj" type="number" value="0" min="0" style="width:80px"/>
Click Prompt Type: <select id="lab"><option>Positive</option><option>Negative</option></select>
</div>
<pre id="status" style="color:green"></pre>
<script>
let idx=0, natW=0, natH=0, key_frames=[];
async function fetch_key_frames(){
  let res = await fetch('/key_frames');
  key_frames = await res.json();
}
function get_true_idx(){
    return key_frames[idx];
}

async function load_state(){
  await fetch_key_frames();
  const obj_id = parseInt(document.getElementById('obj').value || "0");
  let res = await fetch('/state?key_idx='+idx+'&obj_id='+obj_id);
  let js = await res.json();
  idx = js.key_idx;
  document.getElementById('fname').innerText=js.filename+' (Key Frame '+js.key_idx+' / '+(key_frames.length-1)+', Real Index '+js.frame_index+')';
  document.getElementById('img').src=js.image_overlay;
  // Decide which button to use based on the current object
  const saveBtn = document.getElementById('save');
  if(js.mask_saved){
      saveBtn.innerText = "Overwrite";
      saveBtn.style.background = "#ff9800"; // orange
  } else {
      saveBtn.innerText = "Save";
      saveBtn.style.background = "#1976d2"; // blue
  }
  // Automatically update the current frame index when jumping via the input box
  document.getElementById('goto_frame').value = js.key_idx;
}

document.getElementById('obj').addEventListener('input', function(){
    load_state();
});
document.getElementById('goto_btn').onclick = function(){
    let inputIdx = parseInt(document.getElementById('goto_frame').value);
    if(!isNaN(inputIdx) && inputIdx >= 0 && inputIdx < key_frames.length){
        idx = inputIdx;
        load_state();
    }
};

document.getElementById('prev').onclick=()=>{
    if(idx > 0){ idx--; load_state(); }
};
document.getElementById('next').onclick=()=>{
    if(idx < key_frames.length-1){ idx++; load_state(); }
};
document.getElementById('prop').onclick=()=>{document.getElementById('status').innerText='Propagating...'; fetch('/propagate',{method:'POST'}).then(r=>r.json()).then(js=>{document.getElementById('status').innerText=js.message; load_state();});};
document.getElementById('save').onclick=()=>{
    fetch('/save?key_idx='+idx+'&obj_id='+parseInt(document.getElementById('obj').value||"0"),{method:'POST'})
    .then(r=>r.json())
    .then(js=>{
        document.getElementById('status').innerText=js.message;
        const saveBtn = document.getElementById('save');
        const oldBg = saveBtn.style.background;
        saveBtn.style.background = "#43a047"; // green
        saveBtn.innerText = "OK";
        setTimeout(()=>{
            saveBtn.style.background = "#ff9800"; // orange
            saveBtn.innerText = "Overwrite";
        }, 1000);
    });
};
document.getElementById('clear').onclick=()=>{fetch('/clear?key_idx='+idx,{method:'POST'}).then(r=>r.json()).then(js=>{document.getElementById('status').innerText=js.message; load_state();});};
const imgEl=document.getElementById('img');
imgEl.addEventListener('load', ()=>{
  natW = imgEl.naturalWidth;
  natH = imgEl.naturalHeight;
});
imgEl.addEventListener('mousedown', async (e)=>{
  e.preventDefault();
  if (!natW || !natH) {
    natW = imgEl.naturalWidth || imgEl.clientWidth;
    natH = imgEl.naturalHeight || imgEl.clientHeight;
  }
  const rect=imgEl.getBoundingClientRect();
  const scaleX = (natW===0?1:natW/imgEl.clientWidth);
  const scaleY = (natH===0?1:natH/imgEl.clientHeight);
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);
  let label = document.getElementById('lab').value;
  if (e.button===0) label='Positive'; if (e.button===2) label='Negative';
  const payload = {x:x,y:y,label_type:label,obj_id:parseInt(document.getElementById('obj').value||"0"), key_idx: idx};
  const res = await fetch('/click', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const js = await res.json();
  if (js.image_data_url) { imgEl.src = js.image_data_url; }
});
imgEl.addEventListener('contextmenu', (e)=>{ e.preventDefault(); return false; });

// Listen for left/right arrow keys to switch frames
document.addEventListener('keydown', function(event){
    if (event.target.tagName.toLowerCase() === 'input') return; // Prevent accidental triggers during input
    if(event.key === "ArrowLeft"){
        if(idx > 0){ idx--; load_state(); }
    }else if(event.key === "ArrowRight"){
        if(idx < key_frames.length-1){ idx++; load_state(); }
    }
});
window.onload = load_state;
</script>
</body></html>
"""

@app.route("/key_frames")
def key_frames_api():
    return jsonify(key_frame_indices)

@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.route("/state")
def state():
    key_idx = int(request.args.get("key_idx", frame_index))
    obj_id = int(request.args.get("obj_id", 0))  # The default object is object 0
    real_idx = key_frame_indices[key_idx]
    img = load_frame(real_idx)
    overlay_img = current_display_image(real_idx)
    out_path = os.path.join(SAVE_DIR, f"{obj_id:03d}", f"{real_idx:05d}.png")
    has_saved = os.path.exists(out_path)
    return jsonify({
        "key_idx": key_idx,
        "frame_index": real_idx,
        "filename": frame_paths[real_idx],
        "image_original": encode_image_to_data_url(img),
        "image_overlay": encode_image_to_data_url(overlay_img),
        "mask_saved": has_saved
    })

@app.route("/clear", methods=["POST"])
def clear_cur():
    key_idx = int(request.args.get("key_idx", frame_index))
    fi = key_frame_indices[key_idx]
    prompt_cache.pop(fi, None)
    mask_results.pop(fi, None)
    return jsonify({"message": f"Cleared prompts and masks for key frame {fi}."})

@app.route("/nav")
def nav():
    global frame_index
    direction = request.args.get("dir", "next")
    if direction == "prev":
        frame_index = max(0, frame_index - 1)
    else:
        frame_index = min(len(key_frame_indices)-1, frame_index + 1)
    return jsonify({"key_idx": frame_index, "frame_index": key_frame_indices[frame_index], "filename": frame_paths[key_frame_indices[frame_index]]})

@app.route("/click", methods=["POST"])
def click():
    data = request.get_json(force=True)
    x = int(data["x"]); y = int(data["y"])
    label_type = data.get("label_type", "Positive")
    obj_id = int(data.get("obj_id", 0))
    key_idx = int(data.get("key_idx", frame_index))
    fi = key_frame_indices[key_idx]
    prompt_cache.setdefault(fi, {}).setdefault(obj_id, ([], []))
    pts, lbs = prompt_cache[fi][obj_id]
    pts.append([x, y]); lbs.append(1 if label_type=="Positive" else 0)
    points = np.array(pts, dtype=np.float32)
    labels = np.array(lbs, dtype=np.int64)
    try:
        with torch.inference_mode():
            masks, obj_ids, logits = predictor.add_new_points_or_box(inference_state, fi, obj_id, points, labels)
            predictor.propagate_in_video_preflight(inference_state)
    except Exception as e:
        logger.exception("predictor.add_new_points_or_box failed")
        return jsonify({"error": str(e), "image_data_url": encode_image_to_data_url(load_frame(fi))})
    if hasattr(obj_ids, "tolist"):
        try:
            obj_ids = obj_ids.tolist()
        except Exception:
            obj_ids = list(obj_ids)
    pos = None
    try:
        pos = obj_ids.index(obj_id) if obj_id in obj_ids else None
    except Exception:
        try:
            obj_ids_int = [int(o) for o in obj_ids]
            pos = obj_ids_int.index(int(obj_id)) if int(obj_id) in obj_ids_int else None
        except Exception:
            pos = None
    if pos is None:
        pos = -1
    logit = logits[pos]
    mask_np = (logit > 0)
    try:
        mask2d = _ensure_mask_2d_and_match(mask_np)
    except Exception as e:
        logger.exception(f"Mask normalization failed for frame {fi} obj {obj_id}: {e}")
        return jsonify({"error": str(e), "image_data_url": encode_image_to_data_url(load_frame(fi))})
    mask_results.setdefault(fi, {})[int(obj_id)] = mask2d
    img = load_frame(fi)
    if is_key_frame(fi):
        try:
            img_disp = overlay_mask_on_image(img, mask2d, (0,255,0), 0.5)
            return jsonify({"image_data_url": encode_image_to_data_url(img_disp)})
        except Exception as e:
            logger.exception("overlay failed in click endpoint")
            return jsonify({"error": str(e), "image_data_url": encode_image_to_data_url(img)})
    else:
        return jsonify({"image_data_url": encode_image_to_data_url(img)})

@app.route("/propagate", methods=["POST"])
def propagate():
    count = 0
    try:
        with torch.inference_mode():
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if hasattr(out_obj_ids, "tolist"):
                    out_obj_ids = out_obj_ids.tolist()
                for i, oid in enumerate(out_obj_ids):
                    m = out_mask_logits[i]
                    if hasattr(m, "detach"):
                        m = m.detach().cpu().numpy()
                    mask2 = (m > 0)
                    try:
                        mask2 = _ensure_mask_2d_and_match(mask2)
                    except Exception:
                        mask2 = np.squeeze(np.asarray(mask2))
                        h_img, w_img = load_frame(int(out_frame_idx)).shape[:2]
                        if mask2.shape != (h_img, w_img):
                            mask2 = cv2.resize(mask2.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask_results.setdefault(int(out_frame_idx), {})[int(oid)] = mask2
                    count += 1
    except Exception:
        logger.exception("propagation failed")
        return jsonify({"message": "Propagation error (see server log)."})
    return jsonify({"message": f"Propagation completed."})

@app.route("/save", methods=["POST"])
def save_cur():
    key_idx = int(request.args.get("key_idx", frame_index))
    obj_id = int(request.args.get("obj_id", 0))
    fi = key_frame_indices[key_idx]
    h, w = load_frame(fi).shape[:2]
    save_cnt = 0
    # Save only the current object
    m = None
    if fi in mask_results and obj_id in mask_results[fi]:
        m = mask_results[fi][obj_id]
    else:
        # New feature: save an all-zero mask if no mask is available
        m = np.zeros((h, w), dtype=np.uint8)
    out_path = os.path.join(SAVE_DIR, f"{obj_id:03d}", f"{fi:05d}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray((m.astype(np.uint8) * 255)).save(out_path)
    save_cnt += 1
    return jsonify({"message": f"Saved mask for object {obj_id} at key frame {fi} to {out_path}"})

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
