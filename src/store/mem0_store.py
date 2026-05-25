"""Mem0-backed implementation of LangGraph's BaseStore.

This module exposes `Mem0Store`, a drop-in replacement for `InMemoryStore`
that persists long-term memories via Mem0. Nodes interact with it through
the standard `BaseStore` API (`search`, `put`, `get`, `delete`), so Mem0 is
no longer referenced directly from node code.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)

logger = logging.getLogger(__name__)

_client = None


def get_mem0_client():
    """Return a singleton Mem0 client. Cloud if MEM0_API_KEY is set, else local."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("MEM0_API_KEY")
    if api_key:
        from mem0 import MemoryClient
        _client = MemoryClient(api_key=api_key)
    else:
        try:
            from mem0 import Memory
            _client = Memory()
        except Exception as e:
            logger.warning("Failed to initialize local Mem0: %s. Store will be a no-op.", e)
            _client = None

    return _client


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class Mem0Store(BaseStore):
    """BaseStore backed by Mem0.

    Namespace convention: the last element of the namespace tuple is used as
    Mem0's `user_id` (e.g. `("memories", "alice")` → `user_id="alice"`).
    `value` for `put` may contain either a `messages` list (role/content dicts
    forwarded to Mem0's conversation-aware extractor) or a `memory` string.
    Mem0 generates its own IDs, so the `key` supplied to `put` is informational
    only — retrieval happens via semantic search.
    """

    def __init__(self, client=None):
        self._client = client or get_mem0_client()

    # --- Namespace helper -------------------------------------------------

    @staticmethod
    def _user_id(namespace: tuple[str, ...]) -> str:
        return namespace[-1] if namespace else "default"

    # --- BaseStore required surface --------------------------------------

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                results.append(self._put(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                # Mem0 has no native namespace listing; return empty.
                results.append([])
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        # Mem0 SDK is synchronous; passthrough is acceptable for this demo.
        return self.batch(ops)

    # --- Op implementations ----------------------------------------------

    def _get(self, op: GetOp) -> Item | None:
        if self._client is None:
            return None
        try:
            mem = self._client.get(memory_id=op.key)
        except Exception as e:
            logger.warning("Mem0 get failed: %s", e)
            return None
        if not mem:
            return None
        return Item(
            namespace=op.namespace,
            key=op.key,
            value={"memory": mem.get("memory", "")},
            created_at=_parse_ts(mem.get("created_at")),
            updated_at=_parse_ts(mem.get("updated_at", mem.get("created_at"))),
        )

    def _put(self, op: PutOp) -> None:
        if self._client is None:
            return None
        user_id = self._user_id(op.namespace)

        if op.value is None:
            try:
                self._client.delete(memory_id=op.key)
            except Exception as e:
                logger.warning("Mem0 delete failed: %s", e)
            return None

        payload = op.value.get("messages") or op.value.get("memory") or op.value
        try:
            self._client.add(payload, user_id=user_id)
        except Exception as e:
            logger.warning("Mem0 add failed: %s", e)
        return None

    def _search(self, op: SearchOp) -> list[SearchItem]:
        if self._client is None:
            return []
        user_id = self._user_id(op.namespace_prefix)
        try:
            if op.query:
                raw = self._client.search(
                    op.query, user_id=user_id, limit=op.limit + op.offset
                )
            else:
                raw = self._client.get_all(user_id=user_id)
        except Exception as e:
            logger.warning("Mem0 search failed: %s", e)
            return []

        if isinstance(raw, dict):
            raw = raw.get("results", [])
        raw = raw[op.offset : op.offset + op.limit]

        items: list[SearchItem] = []
        for r in raw:
            if not r.get("memory"):
                continue
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key=r.get("id", ""),
                    value={"memory": r["memory"]},
                    created_at=_parse_ts(r.get("created_at")),
                    updated_at=_parse_ts(r.get("updated_at", r.get("created_at"))),
                    score=r.get("score"),
                )
            )
        return items
