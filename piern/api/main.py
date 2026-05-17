"""Unified FastAPI 入口，挂载 synth / training API 与前端静态资源。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from piern.shared.api.audit import install_audit
from piern.shared.api.errors import install_error_handlers
from piern.shared.api.health import router as health_router
from piern.shared.api.routers import integrity, jobs as shared_jobs
from piern.shared.api.security import install_security
from piern.shared.api.static import SPAStaticFiles
from piern.shared.runtime.config import load_runtime_config, log_runtime_config
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


@asynccontextmanager
async def _lifespan(_: FastAPI):
    log_runtime_config()
    yield


def _cors_origins() -> list[str]:
    return list(load_runtime_config().cors_origins)


app = FastAPI(title="PiERN Unified API", version="3.0", lifespan=_lifespan)
install_security(app)
install_error_handlers(app)
install_audit(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _router in [
    health_router,
    shared_jobs.router,
    integrity.router,
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
    app.include_router(_router, prefix="/api")

_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", SPAStaticFiles(directory=str(_dist), html=True), name="static")
