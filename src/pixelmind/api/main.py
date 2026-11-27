from fastapi import FastAPI

from pixelmind import __version__

app = FastAPI(title="PIXELMIND API", version=__version__)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
