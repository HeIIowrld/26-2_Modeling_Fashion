"""FITTA public web gateway.

The public web server runs in the 192.168.0.110 container. Static files are
served locally, while only ``/api`` traffic is sent through a private SSH
tunnel to the scheduled GPU worker.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parent
DEFAULT_GPU_API_URL = "http://127.0.0.1:18000"
PROXY_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
REQUEST_HEADER_BLOCKLIST = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
RESPONSE_HEADERS = {
    "cache-control",
    "content-disposition",
    "content-language",
    "content-type",
    "etag",
    "expires",
    "last-modified",
}


def create_app(
    gpu_api_url: str | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create a gateway app, optionally with an injected transport for tests."""

    upstream = (gpu_api_url or os.environ.get("FITTA_GPU_API_URL") or DEFAULT_GPU_API_URL).rstrip("/")
    timeout = httpx.Timeout(connect=15, read=900, write=900, pool=30)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.gpu_client = httpx.AsyncClient(
            base_url=upstream,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )
        try:
            yield
        finally:
            await application.state.gpu_client.aclose()

    application = FastAPI(
        title="FITTA web gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def proxy_api(request: Request, path: str = "") -> Response:
        body = await request.body()
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in REQUEST_HEADER_BLOCKLIST
        }
        if request.client:
            headers["x-forwarded-for"] = request.client.host
        headers["x-forwarded-host"] = request.headers.get("host", "")
        headers["x-forwarded-proto"] = request.url.scheme

        target = f"/api/{path}" if path else "/api"
        if request.url.query:
            target = f"{target}?{request.url.query}"

        client: httpx.AsyncClient = request.app.state.gpu_client
        try:
            upstream_response = await client.request(
                request.method,
                target,
                content=body,
                headers=headers,
            )
        except httpx.RequestError as error:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "GPU 작업 서버에 연결할 수 없습니다. 잠시 후 다시 시도하세요.",
                    "error": type(error).__name__,
                },
            )

        response_headers = {
            name: value
            for name, value in upstream_response.headers.items()
            if name.lower() in RESPONSE_HEADERS
        }
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    application.add_api_route("/api", proxy_api, methods=PROXY_METHODS)
    application.add_api_route("/api/{path:path}", proxy_api, methods=PROXY_METHODS)

    class NoCacheStaticFiles(StaticFiles):
        def is_not_modified(self, response_headers, request_headers) -> bool:
            return False

        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

    application.mount("/", NoCacheStaticFiles(directory=WEB_DIR / "static", html=True), name="static")
    return application


app = create_app()
