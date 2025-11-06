# Comet Assay - despliegue con Docker

Este proyecto empaqueta la API de segmentación (FastAPI + TensorFlow) y una página web mínima para subir imágenes desde un navegador. El contenedor está preparado para ejecutarse en un servidor remoto y exponer la aplicación en el puerto 8000 por defecto.

## Requisitos

* Docker 24+ (y opcionalmente Docker Compose)
* Modelos U-Net (`.h5`) ubicados dentro de `Model/` (pueden montarse como volumen de solo lectura)
* Carpeta `Input/` y `Comets_Output/` para intercambio de archivos (se crean automáticamente si no existen)

## Construir la imagen

```bash
docker build -t comet-assay-api .
```

## Ejecutar con Docker

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/Model:/app/Model:ro" \
  -v "$PWD/Input:/app/Input" \
  -v "$PWD/Comets_Output:/app/Comets_Output" \
  -v "$PWD/web:/app/web:ro" \
  comet-assay-api
```

La API quedará accesible en `http://<tu-servidor>:8000`. La página web está disponible en la misma URL y consume los endpoints `/models` y `/segment`.

## Docker Compose

```bash
docker compose up --build
```

El archivo `docker-compose.yml` publica el servicio en el puerto 8000 y monta las carpetas locales para preservar entradas/salidas.

## Variables de entorno

* `UVICORN_HOST`: IP de escucha del servidor Uvicorn (por defecto `0.0.0.0`).
* `UVICORN_PORT`: Puerto del proceso FastAPI (por defecto `8000`).
* `MODEL_DIR`, `INPUT_DIR`, `OUTPUT_DIR`, `WEB_DIR`: Rutas dentro del contenedor para modelos, entrada, salidas y estáticos. Ajusta estas variables si modificas la estructura.

## Flujo desde la web

1. Abre `http://<tu-servidor>:8000/` en tu navegador.
2. La lista de modelos se rellena automáticamente desde `/models`.
3. Sube una imagen (`.tif`, `.png`, `.jpg`, …) y pulsa **Segmentar**.
4. El resultado se muestra en pantalla y puede descargarse como PNG.

Las máscaras generadas también se guardan en `Comets_Output/` dentro del contenedor (y en la carpeta enlazada en el host si montaste el volumen).
