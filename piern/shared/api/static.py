from pathlib import Path as FilePath

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


class SPAStaticFiles(StaticFiles):
    """为 BrowserRouter 提供 history fallback，同时避免吞掉 API 和静态资源的 404。"""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            if path.startswith('api/'):
                raise
            if FilePath(path).suffix:
                raise
            return await super().get_response('index.html', scope)
