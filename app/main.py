import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import SAM_MODEL_NAME, analyze

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LeafScan API", description="Análise de área saudável vs contaminada em folhas")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/sw.js")
def service_worker():
    return FileResponse("static/sw.js", media_type="text/javascript")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": SAM_MODEL_NAME}


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="arquivo vazio")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="arquivo de imagem inválido")
    try:
        result = analyze(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(result)
