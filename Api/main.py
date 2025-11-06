from pathlib import Path
import os

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
    FileResponse,
)
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, cv2, io

from .inference import run_inference, list_models, save_mask, load_unet

HERE = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("BASE_DIR", HERE.parent))
WEB_DIR = Path(os.getenv("WEB_DIR", BASE_DIR / "web"))
INDEX_FILE = WEB_DIR / "index.html"

app = FastAPI(title="Comet Assay API")

# CORS: ajusta allow_origins cuando tengas tu dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["POST","GET"], allow_headers=["*"],
)

@app.on_event("startup")
def _warm():
    # precarga el modelo por defecto si está disponible
    try:
        load_unet(None)
    except FileNotFoundError as exc:
        print(f"[startup] Advertencia: {exc}")


@app.get("/")
def index():
    if INDEX_FILE.is_file():
        return FileResponse(INDEX_FILE)
    return PlainTextResponse(
        "Frontend no disponible. Copia los archivos estáticos en 'web/'.",
        status_code=503,
    )

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/models")
def models():
    return {"models": list_models()}

@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    model: str | None = Query(default=None, description="Nombre del .h5 (opcional)")
):
    data = await image.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Imagen inválida"}, status_code=400)

    model_name = model or None

    try:
        mask = run_inference(img, model_name=model_name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    # Guarda también en Comets_Output con el mismo nombre + _out.png
    basename = os.path.splitext(image.filename or "image.png")[0]
    out_name = f"{basename}_output.png"
    save_mask(mask, out_name)

    ok, buf = cv2.imencode(".png", mask)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
