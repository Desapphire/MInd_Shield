"""
Simple event bus for local async-friendly pipeline wiring.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, List


class EventBus:
    def __init__(self):
        self._handlers: DefaultDict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def on(self, event_name: str, handler: Callable[[Any], None]) -> None:
        self._handlers[event_name].append(handler)

    def emit(self, event_name: str, payload: Any) -> None:
        for handler in list(self._handlers.get(event_name, [])):
            try:
                handler(payload)
            except Exception:
                pass
