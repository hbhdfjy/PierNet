"""Unified FastAPI 入口，挂载 synth / training API 与前端静态资源。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from piern.shared.api.static import SPAStaticFiles
from piern.shared.runtime.paths import PROJECT_ROOT
from piern.synth.api.routers import (
    config,
    datasets,
    files,
    file_catalog,
    generation,
    interview,
    jobs,
    registry,
    router_data,
    simulation,
)
from piern.training.api.routers import training

app = FastAPI(title='PiERN Unified API', version='3.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'http://localhost:4173',
        'http://127.0.0.1:5173',
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

for _router in [
    datasets.router,
    config.router,
    registry.router,
    jobs.router,
    generation.router,
    files.router,
    file_catalog.router,
    interview.router,
    simulation.router,
    router_data.router,
    training.router,
]:
    app.include_router(_router, prefix='/api')

_dist = PROJECT_ROOT / 'frontend' / 'dist'
if _dist.exists():
    app.mount('/', SPAStaticFiles(directory=str(_dist), html=True), name='static')
