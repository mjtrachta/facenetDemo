#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FaceNet + MTCNN + (opcional) Anti-spoof InsightFace
- Reconocimiento por similitud coseno
- Liveness (InsightFace) sobre la CARA ALINEADA (MTCNN)
- Heurística pasiva anti-replay (pantalla rotada + líneas)
- Microtexturas (FFT + HighFreq + LBP, sólo en piel, con tolerancia a blur)
- HTTP (POST /reconocer), WebSocket (/ws/reconocer), Webcam local
"""

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import sys
import time

# FastAPI
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn

# LBP
from skimage.feature import local_binary_pattern

# ---------------- Configuración global ----------------
CARPETA_PERSONAS = Path("/home/hari1/Pictures/Camera")
UMBRAL_SIMILITUD = 0.80
UMBRAL_LIVENESS  = 0.95
EXTS = {'.jpg', '.jpeg', '.png'}

# Heurísticas anti-replay (ajustables)
ENABLE_SCREEN_HEURISTIC = True
MAX_FACE_FRAC = 0.60
RECT_AREA_MIN_FRAC = 0.07
RECT_ASPECT_MIN, RECT_ASPECT_MAX = 1.2, 3.2
RECT_IOU_MIN = 0.20
CONTEXT_MARGIN = 0.80

# Microtexturas (ajustables)
ENABLE_MICROTEXTURE = True
LBP_POINTS, LBP_RADIUS = 8, 1
LBP_METHOD = "uniform"
# Umbrales afinados para condiciones reales (webcam/indoor)
LBP_ENTROPY_MIN = 1.3
HF_ENERGY_MIN   = 0.12
FFT_PEAK_MAX    = 2.4
# Tolerancia a blur: si la nitidez es baja, no se castiga LBP
BLUR_VAR_MIN    = 80.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Dispositivo usado: {device}")

# Flags / modelos globales
ENABLE_LIVENESS = True
mtcnn = None
resnet = None
embeddings = None
nombres = None
spoof = None  # anti-spoof de insightface

# -------- Utilidades anti-replay (heurísticas) --------
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def looks_like_phone_screen_any_angle(context_bgr: np.ndarray,
                                      face_bbox_full: Tuple[int,int,int,int],
                                      ctx_origin_xy: Tuple[int,int]) -> bool:
    """
    Busca pantalla de celular/tablet (rotada o no) combinando:
      - Contornos → cuadriláteros grandes (minAreaRect) con AR de pantalla
      - HoughLines → dos familias de líneas casi ortogonales (rejilla/bordes)
    """
    if context_bgr is None or context_bgr.size == 0:
        return False

    Hc, Wc = context_bgr.shape[:2]
    gray = cv2.cvtColor(context_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fx1, fy1, fx2, fy2 = face_bbox_full
    ox, oy = ctx_origin_xy
    has_rect = False
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < 80:
            continue
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area < RECT_AREA_MIN_FRAC * (Hc * Wc):
                continue
            rect = cv2.minAreaRect(approx)
            (cx, cy), (rw, rh), _theta = rect
            if rw < 1 or rh < 1:
                continue
            ar = max(rw, rh) / max(1.0, min(rw, rh))
            if not (RECT_ASPECT_MIN <= ar <= RECT_ASPECT_MAX):
                continue
            box = cv2.boxPoints(rect).astype(int)
            bx1, by1 = box[:,0].min() + ox, box[:,1].min() + oy
            bx2, by2 = box[:,0].max() + ox, box[:,1].max() + oy
            if iou((fx1,fy1,fx2,fy2), (bx1,by1,bx2,by2)) >= RECT_IOU_MIN:
                has_rect = True
                break

    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=120)
    grid_like = False
    if lines is not None and len(lines) >= 8:
        thetas = (np.rad2deg(lines[:,0,1]) % 180.0)
        hist, bins = np.histogram(thetas, bins=np.arange(0, 181, 10))
        if hist.sum() > 0:
            top2 = hist.argsort()[-2:]
            ang1 = (bins[top2[0]] + bins[top2[0]+1]) * 0.5
            ang2 = (bins[top2[1]] + bins[top2[1]+1]) * 0.5
            dist = min(abs(ang1-ang2), 180-abs(ang1-ang2))
            if hist[top2[0]] >= 4 and hist[top2[1]] >= 4 and 70 <= dist <= 110:
                grid_like = True

    return has_rect or grid_like

def bbox_too_big(x1:int, y1:int, x2:int, y2:int, W:int, H:int, max_frac:float=MAX_FACE_FRAC) -> bool:
    bw, bh = (x2 - x1) / max(1, W), (y2 - y1) / max(1, H)
    return (bw >= max_frac) or (bh >= max_frac)

# -------- Microtexturas --------
def skin_mask_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Máscara binaria de piel (0/1) usando YCrCb + limpieza morfológica.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return (mask > 0).astype(np.uint8)

def tensor_to_bgr224(face_torch: torch.Tensor) -> np.ndarray:
    face_np = face_torch.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    face_np = np.clip(face_np * 255.0, 0, 255).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)

def microtexture_metrics(face_bgr224: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Devuelve (fft_peak_ratio, hf_energy, lbp_entropy, blur_var).
    - FFT: global en 224x224 (picos periódicos por rejilla de pantalla)
    - HF y LBP: sólo en piel
    - blur_var: varianza del Laplaciano (nitidez); bajo = blur
    """
    gray = cv2.cvtColor(face_bgr224, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # Blur (nitidez)
    blur_var = float(cv2.Laplacian(gray_eq, cv2.CV_32F).var())

    # FFT (magnitud, bandas medias) – global
    F = np.fft.fft2(gray_eq)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    h, w = gray_eq.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    Rn = R / (R.max() + 1e-6)

    edges = np.linspace(0.15, 0.45, 8)
    vals = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (Rn >= a) & (Rn < b)
        vals.append(mag[m].mean() if m.sum() else 0.0)
    vals = np.array(vals) + 1e-9
    fft_peak_ratio = float(vals.max() / vals.mean())

    # Máscara de piel
    skin = skin_mask_bgr(face_bgr224)
    if int(skin.sum()) < 300:
        skin = np.ones_like(skin, dtype=np.uint8)

    # High-frequency energy SOLO en piel (Laplaciano)
    lap = cv2.Laplacian(gray_eq, cv2.CV_32F)
    hf_energy = float(np.mean(np.abs(lap)[skin > 0])) / 255.0

    # LBP SOLO en piel
    lbp = local_binary_pattern(gray_eq, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    lbp_skin = lbp[skin > 0]
    hist, _ = np.histogram(lbp_skin.ravel(),
                           bins=np.arange(0, LBP_POINTS+3),
                           range=(0, LBP_POINTS+2),
                           density=True)
    lbp_entropy = float(-(hist * np.log2(hist + 1e-12)).sum())

    return fft_peak_ratio, hf_energy, lbp_entropy, blur_var

def microtexture_ok(face_bgr224: np.ndarray) -> Tuple[bool, Tuple[float,float,float,float]]:
    """
    Regla 2-de-3 + excepción por blur:
      - Si blur_var < BLUR_VAR_MIN → ignoramos LBP (tiende a bajar con desenfoque)
      - Pasan al menos 2 de {FFT, HF, LBP} → ok
    """
    fft_peak, hf_e, lbp_ent, blur_var = microtexture_metrics(face_bgr224)

    pass_fft = (fft_peak <= FFT_PEAK_MAX)
    pass_hf  = (hf_e     >= HF_ENERGY_MIN)
    pass_lbp = (lbp_ent  >= LBP_ENTROPY_MIN)

    if blur_var < BLUR_VAR_MIN:
        pass_count = int(pass_fft) + int(pass_hf)
    else:
        pass_count = int(pass_fft) + int(pass_hf) + int(pass_lbp)

    ok = (pass_count >= 2)
    return ok, (fft_peak, hf_e, lbp_ent, blur_var)

# -------- Inicializar modelo y enrolar personas --------
def init_model():
    global mtcnn, resnet, embeddings, nombres, spoof, ENABLE_LIVENESS
    from facenet_pytorch import MTCNN, InceptionResnetV1
    mtcnn = MTCNN(image_size=160, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    try:
        import onnxruntime as ort
        from insightface.model_zoo import get_model
        avail = ort.get_available_providers()
        providers = [ep for ep in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if ep in avail]
        spoof = get_model("buffalo_l_antispoof", providers=providers)
        print(f"🛡️  Anti-spoof cargado (providers={providers})")
    except Exception:
        ENABLE_LIVENESS = False
        spoof = None
        print("⚠️  Liveness deshabilitado (insightface/onnxruntime no disponible). "
              "Para habilitarlo: pip install insightface onnxruntime(-gpu)")

    embs_list, names = [], []
    print("🔍 Enrolando personas desde:", CARPETA_PERSONAS)
    for carpeta in sorted(CARPETA_PERSONAS.iterdir()):
        if not carpeta.is_dir(): continue
        nombre, embs = carpeta.name, []
        for img_path in carpeta.iterdir():
            if img_path.suffix.lower() not in EXTS: continue
            try: img = Image.open(img_path).convert("RGB")
            except Exception: continue
            face = mtcnn(img)
            if face is None: continue
            if face.ndim == 3: face = face.unsqueeze(0)
            with torch.no_grad():
                emb = resnet(face.to(device))
            for i in range(emb.shape[0]):
                embs.append(emb[i].detach())
        if embs:
            prom = torch.stack(embs).mean(0)
            prom = prom / prom.norm()
            embs_list.append(prom)
            names.append(nombre)
            print(f"✓ {nombre}: {len(embs)} rostros enrolados")
        else:
            print(f"⚠ {nombre}: sin rostros válidos, omitido")
    if not embs_list:
        raise RuntimeError("❌ No se enroló ninguna persona.")
    globals()['embeddings'] = torch.stack(embs_list).to(device)
    globals()['nombres'] = names
    print(f"🎉 Enroladas {len(nombres)} personas: {', '.join(nombres)}")

# -------- Estructura de resultado --------
class ReconResult(BaseModel):
    nombre: str
    similitud: float
    bbox: List[int]
    liveness: bool
    live_score: float
    micro_ok: bool
    fft_peak: float
    hf_energy: float
    lbp_entropy: float
    blur_var: float

# -------- Liveness desde cara alineada (MTCNN) --------
def check_liveness_from_tensor(face_torch: torch.Tensor) -> Tuple[bool, float]:
    global spoof, ENABLE_LIVENESS
    if not ENABLE_LIVENESS or spoof is None:
        return True, 1.0
    face_bgr = tensor_to_bgr224(face_torch)
    try:
        out = spoof(face_bgr)
        if isinstance(out, tuple) and len(out) == 2:
            is_live, score = bool(out[0]), float(out[1])
        elif isinstance(out, dict):
            score = float(out.get("score", 0.0))
            is_live = bool(out.get("is_live", score >= UMBRAL_LIVENESS))
        else:
            is_live, score = False, 0.0
        return (is_live and score >= UMBRAL_LIVENESS), score
    except Exception:
        return False, 0.0

# -------- Función principal de reconocimiento --------
def reconocer_frame(frame: np.ndarray) -> List[ReconResult]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)

    faces_tensor = mtcnn(img_pil)
    boxes, _ = mtcnn.detect(img_pil)
    if faces_tensor is None or boxes is None:
        return []
    if faces_tensor.ndim == 3:
        faces_tensor = faces_tensor.unsqueeze(0)

    results = []
    H, W = frame.shape[:2]
    n = min(faces_tensor.shape[0], len(boxes))
    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)

        # Cara alineada
        face_t = faces_tensor[i:i+1].to(device)
        face_bgr224 = tensor_to_bgr224(face_t)

        # Liveness
        is_live, live_score = check_liveness_from_tensor(face_t)

        # Microtexturas
        micro_ok, (fft_peak, hf_energy, lbp_entropy, blur_var) = microtexture_ok(face_bgr224)

        # Heurísticas anti-replay en contexto grande
        suspicious = False
        if bbox_too_big(x1c, y1c, x2c, y2c, W, H, MAX_FACE_FRAC):
            suspicious = True
        if ENABLE_SCREEN_HEURISTIC:
            bw, bh = (x2c - x1c), (y2c - y1c)
            mx, my = int(bw * CONTEXT_MARGIN), int(bh * CONTEXT_MARGIN)
            X1, Y1 = max(0, x1c - mx), max(0, y1c - my)
            X2, Y2 = min(W, x2c + mx), min(H, y2c + my)
            context = frame[Y1:Y2, X1:X2]
            if looks_like_phone_screen_any_angle(context, (x1c,y1c,x2c,y2c), (X1,Y1)):
                suspicious = True

        passed_gate = (is_live and micro_ok and not suspicious)

        nombre = "Desconocido"
        sim = 0.0
        if passed_gate:
            with torch.no_grad():
                emb_live = resnet(face_t)
                emb_live = emb_live / emb_live.norm(dim=1, keepdim=True)
            sim_mat = torch.nn.functional.cosine_similarity(
                emb_live.unsqueeze(1), embeddings.unsqueeze(0), dim=2
            )
            best_idx = int(torch.argmax(sim_mat, dim=1).item())
            best_sim = float(torch.max(sim_mat, dim=1).values.item())
            if best_sim > UMBRAL_SIMILITUD:
                nombre, sim = nombres[best_idx], best_sim
        else:
            nombre = "FAKE"

        results.append(ReconResult(
            nombre=nombre,
            similitud=sim,
            bbox=[x1, y1, x2, y2],
            liveness=bool(is_live),
            live_score=float(live_score),
            micro_ok=bool(micro_ok),
            fft_peak=float(fft_peak),
            hf_energy=float(hf_energy),
            lbp_entropy=float(lbp_entropy),
            blur_var=float(blur_var),
        ))
    return results

# -------- Configurar FastAPI --------
app = FastAPI(title="FaceNet + (Opcional) Liveness API")

@app.post("/reconocer", response_model=List[ReconResult])
async def reconocer_imagen(file: UploadFile = File(...)):
    contenido = await file.read()
    nparr = np.frombuffer(contenido, dtype=np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return reconocer_frame(frame) if frame is not None else []

@app.websocket("/ws/reconocer")
async def ws_reconocer(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            results = reconocer_frame(frame)
            await websocket.send_json([r.dict() for r in results])
    except WebSocketDisconnect:
        pass

# -------- Modo webcam --------
def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara.")
    print("🎥 Reconocimiento" + (" + Liveness" if ENABLE_LIVENESS else "") + " — pulsa 'q' para salir")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tic = time.time()
        results = reconocer_frame(frame)
        for r in results:
            x1, y1, x2, y2 = r.bbox
            if (ENABLE_LIVENESS and not r.liveness) or (not r.micro_ok) or r.nombre == "FAKE":
                color = (0, 165, 255)  # naranja
                label = f"FAKE {r.live_score:.2f} | mt p:{r.fft_peak:.2f} hf:{r.hf_energy:.2f} h:{r.lbp_entropy:.2f} b:{r.blur_var:.0f}"
            else:
                reconocido = (r.nombre != "Desconocido" and r.nombre != "FAKE")
                color = (0, 255, 0) if reconocido else (0, 0, 255)
                base = f"{r.nombre}" if reconocido else "Desconocido"
                label = f"{base} {r.similitud:.2f} | live {r.live_score:.2f} | mt ok"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        fps = 1.0 / max(1e-6, (time.time() - tic))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Reconocimiento Facial", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# -------- Inicio principal --------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("Selecciona modo [webcam/serve/ws]: ").strip().lower()

    init_model()

    if mode == "webcam":
        run_webcam()
    elif mode in ("serve", "ws"):
        uvicorn.run("main:app", host="0.0.0.0", port=8000)
    else:
        print(f"Modo desconocido: {mode}")
