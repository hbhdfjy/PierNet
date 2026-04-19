from pathlib import Path as FilePath

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


class SPAStaticFiles(StaticFiles):
    """? BrowserRouter ?? history fallback??? API ?????? 404?"""

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
