"""Unified FastAPI 入口，挂载 synth / training API 与前端静态资源。"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from piern.shared.api.errors import install_error_handlers
from piern.shared.api.health import router as health_router
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


def _cors_origins() -> list[str]:
    configured = os.getenv('PIERN_CORS_ORIGINS', '').strip()
    if configured:
        return [item.strip() for item in configured.split(',') if item.strip()]
    frontend_port = os.getenv('PIERN_FRONTEND_PORT', '5173')
    return [
        f'http://localhost:{frontend_port}',
        'http://localhost:4173',
        f'http://127.0.0.1:{frontend_port}',
    ]


app = FastAPI(title='PiERN Unified API', version='3.0')
install_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

for _router in [
    health_router,
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
