"""FastAPI 应用入口：注册所有 router，配置 CORS 和静态文件。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from piern.api.deps import PROJECT_ROOT
from piern.api.routers import (
    datasets,
    config,
    registry,
    jobs,
    generation,
    files,
    interview,
    simulation,
)

app = FastAPI(title="PiERN 多模拟器数据集 API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册所有路由（统一前缀 /api）
for _router in [
    datasets.router,
    config.router,
    registry.router,
    jobs.router,
    generation.router,
    files.router,
    interview.router,
    simulation.router,
]:
    app.include_router(_router, prefix="/api")

# 生产模式：挂载前端静态文件
_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
