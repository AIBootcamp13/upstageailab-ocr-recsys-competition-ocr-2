# Docker helper and dev image

Quick notes:

- First build can be large and slow: many CUDA-enabled wheels and native extensions are downloaded and compiled.
- To build and run GPU dev container (in WSL):

```bash
cd /path/to/repo
# Build and start GPU dev container
docker compose -f docker/docker-compose.yml --profile dev up -d --build
```

- To open a shell inside the container:

```bash
docker exec -it ocr-dev /bin/bash
```

- If you hit build failures for native extensions, consider adding required system packages to `docker/Dockerfile` or running `uv sync` inside the container to debug.

- To stop and remove containers and volumes:

```bash
# Stop and remove containers, networks, volumes
docker compose -f docker/docker-compose.yml down -v
```
