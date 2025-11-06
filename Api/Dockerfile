FROM condaforge/miniforge3:24.3.0-0
SHELL ["/bin/bash", "-lc"]

WORKDIR /app

# Copia el entorno y lo crea con conda
COPY environment.yml /app/
RUN conda env create -f environment.yml
ENV PATH=/opt/conda/envs/comet310/bin:$PATH

# Copia el código del proyecto
COPY UNET_COMET_ASSAY /app/UNET_COMET_ASSAY
COPY api /app/api

EXPOSE 8000
ENV UVICORN_WORKERS=2
CMD uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}
