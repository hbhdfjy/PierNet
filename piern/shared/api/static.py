from pathlib import Path as FilePath

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


def _disable_html_cache(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class SPAStaticFiles(StaticFiles):
    """为 BrowserRouter 提供 history fallback，同时避免吞掉 API 和静态资源的 404。"""

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
            if path in {"", ".", "index.html"}:
                return _disable_html_cache(response)
            return response
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            if path == 'api' or path.startswith('api/'):
                raise
            if FilePath(path).suffix:
                raise
            return _disable_html_cache(await super().get_response('index.html', scope))
