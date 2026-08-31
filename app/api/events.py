"""In-process Asyncio Event Bus for Realtime UI Updates."""
import asyncio
from typing import Any, Callable, Dict, List, Set
from loguru import logger


class EventBus:
    """Lightweight in-process Pub/Sub event bus."""

    def __init__(self):
        self._subscribers: Dict[str, Set[Callable[[Any], Any]]] = {}

    def subscribe(self, topic: str, callback: Callable[[Any], Any]) -> None:
        """Subscribe a callback to a topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = set()
        self._subscribers[topic].add(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Any], Any]) -> None:
        """Unsubscribe a callback."""
        if topic in self._subscribers:
            self._subscribers[topic].discard(callback)

    def publish(self, topic: str, data: Any) -> None:
        """Publish data to all topic subscribers."""
        callbacks = self._subscribers.get(topic, set()).copy()
        for cb in callbacks:
            try:
                res = cb(data)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.error(f"Error invoking callback for topic {topic}: {e}")


# Global singleton instance
event_bus = EventBus()
