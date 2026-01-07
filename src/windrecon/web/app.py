import io
import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from windrecon.inference import load_checkpoint, reconstruct

app = FastAPI(title="WindRecon Web")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

templates = Jinja2Templates(directory=TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model_cache = {}


def _get_model(model_path: str):
    if model_path not in model_cache:
        model_cache[model_path] = load_checkpoint(model_path)
    return model_cache[model_path]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/infer")
async def infer(
    file: UploadFile = File(...),
    model_path: str = Form(...),
    index: int = Form(0),
    mc: int = Form(1),
):
    try:
        content = await file.read()
        data = np.load(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read .npz: {exc}")

    required_keys = {"features", "winds", "lengths", "altitude_bins"}
    if not required_keys.issubset(set(data.files)):
        raise HTTPException(status_code=400, detail=f".npz missing keys. Required: {required_keys}")

    features = data["features"]
    winds = data["winds"]
    lengths = data["lengths"]
    altitude_bins = data["altitude_bins"]

    if index < 0 or index >= features.shape[0]:
        raise HTTPException(status_code=400, detail=f"Index {index} out of range (0..{features.shape[0]-1})")

    model = _get_model(model_path)
    device = torch.device("cpu")
    model.to(device)

    pred, std = reconstruct(model, features[index], int(lengths[index]), altitude_bins, mc_samples=mc)
    result = []
    for i, alt in enumerate(altitude_bins):
        true_u, true_v = winds[index, i]
        result.append(
            {
                "altitude_m": float(alt),
                "wind_u_mps": float(pred[i, 0]),
                "wind_v_mps": float(pred[i, 1]),
                "std_u": float(std[i, 0]),
                "std_v": float(std[i, 1]),
                "true_u": float(true_u),
                "true_v": float(true_v),
            }
        )
    return JSONResponse({"profile": result})


@app.get("/health")
async def health():
    return {"status": "ok"}
