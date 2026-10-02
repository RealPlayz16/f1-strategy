"""Pit-wall dashboard: replay a holdout race and show the live decision at each call.

Serves the page in dashboard/ and the committed snapshots in data/demo/ (written by
src/dashboard_data.py). Every decision shown was computed by src.live.decide from the race
state as of that moment, so the page cannot show a decision that used the future.

Usage:
    python -m src.dashboard               (http://127.0.0.1:8000)
    python -m src.dashboard --port 8001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

DEMO_DIR = Path("data/demo")
PAGE_DIR = Path("dashboard")

app = FastAPI(title="F1 strategy pit wall", docs_url=None, redoc_url=None)


@app.get("/api/races")
def races() -> JSONResponse:
    path = DEMO_DIR / "index.json"
    if not path.exists():
        raise HTTPException(503, "no demo data: run python -m src.dashboard_data")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/race/{rid}")
def race(rid: str) -> JSONResponse:
    path = DEMO_DIR / f"{rid}.json"
    if not path.resolve().is_relative_to(DEMO_DIR.resolve()) or not path.exists():
        raise HTTPException(404, f"unknown race {rid}")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PAGE_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(PAGE_DIR)), name="static")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the pit-wall dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not (DEMO_DIR / "index.json").exists():
        print("no demo data found. Run: python -m src.dashboard_data", file=sys.stderr)
        return 1
    import uvicorn

    print(f"pit wall on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
