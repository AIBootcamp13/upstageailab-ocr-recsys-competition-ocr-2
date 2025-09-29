Maintenance log for docker environment

Status: In progress

Actions performed:

- Hardened `docker/Dockerfile` with additional system packages to build native extensions (cmake, pkg-config, libjpeg-dev, libpng-dev, libsndfile1-dev, ffmpeg, libgl1-mesa-glx, python3-dev, libpython3-dev).
- Adjusted `docker/docker-compose.yml` build contexts to use repository root so Dockerfile COPY can access repo files.
- Made `uv sync` conditional during build and ensured `README.md` is copied into the image.
- Built and started the GPU `dev` container; container `ocr-dev` is running.

Planned next steps:
- Run in-container smoke tests and append results here.
- Add `docker/README.md` with quick start notes.
- Optionally stop/remove containers to free space.

Smoke tests (to be filled):

- torch.cuda.is_available():
- uv availability:
- python version and basic import checks:

Build notes:
- The first build downloads many large CUDA-related wheels (torch, cudnn, etc.) and native libs; expect long first-build times.
- Image base: `nvidia/cuda:12.8.1-devel-ubuntu22.04`.

Commands used:
- docker compose -f docker/docker-compose.yml --profile dev up -d --build
- docker ps -a --filter "name=ocr-dev"
- docker exec -it ocr-dev /bin/bash

Smoke test results (executed inside container ocr-dev):

- PYTHON: /usr/bin/python3 3.10.12
- TORCH_IMPORT_ERROR: No module named 'torch' (torch not installed in the uv environment)
- UV_VERSION: uv 0.8.22

Notes:
- At build time we ran `uv sync --frozen` which attempted to install torch and other packages; torch wasn't available in the runtime uv environment (possible reasons: installation in build cache, or packages installed as part of build but not present in runtime PATH); we may need to run `uv sync` at container runtime or ensure wheels are cached properly.
