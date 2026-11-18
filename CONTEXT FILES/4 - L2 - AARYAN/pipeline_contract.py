"""Small adapter contract between the API and the team-owned matcher."""

from __future__ import annotations

import importlib
import os
from typing import Any, Protocol


class MatchEngine(Protocol):
    def is_ready(self) -> bool:
        """Return true only when the trained model and decoder are loaded."""

    def match_trip(
        self,
        trip_id: str,
        track: list[list[float]],
        nodes: list[list[float]],
        arcs: list[list[int]],
    ) -> dict[str, Any]:
        """Return the predicted route and only evaluator-safe metrics/geometry."""


def load_engine() -> MatchEngine:
    """Load a team integration module selected through PATHSNAP_ENGINE_MODULE.

    That module must expose create_engine() and return an object implementing
    MatchEngine. Keeping this adapter small lets Akshay and Abhiroop finish
    their feature and decoder interfaces before the final wiring is chosen.
    """
    module_name = os.environ.get("PATHSNAP_ENGINE_MODULE")
    if not module_name:
        raise RuntimeError("PATHSNAP_ENGINE_MODULE is not configured yet.")
    module = importlib.import_module(module_name)
    if not hasattr(module, "create_engine"):
        raise RuntimeError(f"{module_name} must define create_engine().")
    engine = module.create_engine()
    if not callable(getattr(engine, "is_ready", None)):
        raise RuntimeError("The matching engine must define is_ready().")
    if not callable(getattr(engine, "match_trip", None)):
        raise RuntimeError("The matching engine must define match_trip().")
    return engine
