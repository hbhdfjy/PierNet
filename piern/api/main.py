"""FastAPI ????????? router??? CORS ??????"""

from pathlib import Path as FilePath

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    router_data,
)


class SPAStaticFiles(StaticFiles):
    """? BrowserRouter ?? history fallback??????? 404?"""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            # ??????????????????? 404
            if FilePath(path).suffix:
                raise
            return await super().get_response('index.html', scope)


app = FastAPI(title="PiERN ??????? API", version="2.0")

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

# ??????????? /api?
for _router in [
    datasets.router,
    config.router,
    registry.router,
    jobs.router,
    generation.router,
    files.router,
    interview.router,
    simulation.router,
    router_data.router,
]:
    app.include_router(_router, prefix="/api")

# ?????????????
_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", SPAStaticFiles(directory=str(_dist), html=True), name="static")
