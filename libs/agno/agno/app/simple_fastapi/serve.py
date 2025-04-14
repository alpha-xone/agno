from typing import Union
from urllib.parse import quote

from fastapi import FastAPI
from rich import box
from rich.panel import Panel

from agno.api.playground import PlaygroundEndpointCreate, create_playground_endpoint
from agno.cli.console import console
from agno.cli.settings import agno_cli_settings
from agno.utils.log import logger


def serve_api_app(
    app: Union[str, FastAPI],
    *,
    host: str = "localhost",
    port: int = 7777,
    reload: bool = False,
    **kwargs,
):
    import uvicorn

    logger.info(f"Starting API on {host}:{port}")

    uvicorn.run(app=app, host=host, port=port, reload=reload, **kwargs)
