from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, cv2, io, os
from .inference import run_inference, list_models, save_mask, load_unet

app = FastAPI(title="Comet Assay API")

# CORS: ajusta allow_origins cuando tengas tu dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["POST","GET"], allow_headers=["*"],
)

@app.on_event("startup")
def _warm():
    # precarga el modelo por defecto
    load_unet(None)

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

    try:
        mask = run_inference(img, model_name=model)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    # Guarda también en Comets_Output con el mismo nombre + _out.png
    basename = os.path.splitext(image.filename or "image.png")[0]
    out_name = f"{basename}_output.png"
    save_mask(mask, out_name)

    ok, buf = cv2.imencode(".png", mask)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")