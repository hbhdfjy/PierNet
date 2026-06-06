"""Unified FastAPI 入口，挂载 synth / training API 与前端静态资源。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from PierNet.shared.api.audit import install_audit
from PierNet.shared.api.errors import install_error_handlers
from PierNet.shared.api.health import router as health_router
from PierNet.shared.api.routers import metrics
from PierNet.shared.api.static import SPAStaticFiles
from PierNet.shared.runtime.config import load_runtime_config, log_runtime_config
from PierNet.shared.runtime.paths import PROJECT_ROOT
from PierNet.synth.api.routers import (
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
from PierNet.training.api.routers import assembly, text2comp, training


@asynccontextmanager
async def _lifespan(_: FastAPI):
    simulation.cleanup_stale_tmp_configs()
    log_runtime_config()
    yield


def _cors_origins() -> list[str]:
    return list(load_runtime_config().cors_origins)


app = FastAPI(title="PierNet Unified API", version="3.0", lifespan=_lifespan)
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
    metrics.router,
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
    text2comp.router,
    assembly.router,
    training.router,
]:
    app.include_router(_router, prefix="/api")

_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", SPAStaticFiles(directory=str(_dist), html=True), name="static")
