"""Logfire observability for the Stellabook pipeline."""

import os

import logfire
from fastapi import FastAPI


def configure_observability(app: FastAPI) -> None:
    """Configure Logfire and instrument FastAPI."""
    logfire.configure(
        environment=os.environ.get("LOGFIRE_ENVIRONMENT", "local"),
    )
    logfire.instrument_fastapi(app)
