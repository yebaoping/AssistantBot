import json
import logging
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from mem0 import Memory
from mem0.memory.main import _build_session_scope
from pydantic import BaseModel
import redis


logger = logging.getLogger(__name__)

config = {
    # "history_db_path": "./history.db",
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": os.getenv("QDRANT_HOST"),
            "port": os.getenv("QDRANT_PORT"),
            "collection_name": "assistant_bot",
            "embedding_model_dims": 1024,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("OLLAMA_MODEL"),
            "temperature": 0.1,
            "ollama_base_url": os.getenv("OLLAMA_HOST"),
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": os.getenv("VLLM_EMBEDDING_MODEL"),
            "openai_base_url": os.getenv("VLLM_EMBEDDING_HOST"),
        },
    },
    "reranker": {
        "provider": "llm_reranker",
        "config": {
            "provider": "openai",
            "model": os.getenv("VLLM_RERANKER_MODEL"),
            "openai_base_url": os.getenv("VLLM_RERANKER_HOST"),
        },
    },
}

memory = Memory.from_config(config)


pool = redis.ConnectionPool.from_url(os.getenv("REDIS_URL"))
redis_client = redis.Redis(connection_pool=pool)


class RunHistoryItem(BaseModel):
    run_id: str
    user_id: str
    content: str


def _get_user_run_id_key(user_id: str) -> str:
    return f"assistant_bot_user_run_id:{user_id}"


def _get_user_run_id_content_key(user_id: str) -> str:
    return f"assistant_bot_user_run_id_content:{user_id}"


def add_user_run_id(user_id: str, run_id: str, content: str | list) -> None:
    key = _get_user_run_id_key(user_id)

    if redis_client.zscore(key, run_id) is not None:
        return

    if isinstance(content, list):
        content = json.dumps(content)

    redis_client.zadd(key, {run_id: time.time_ns()})
    redis_client.hset(_get_user_run_id_content_key(user_id), run_id, content)


def run_history(user_id: str) -> list[RunHistoryItem]:
    result = []

    key = _get_user_run_id_key(user_id)

    run_ids = redis_client.zrevrange(key, 0, 20)
    for run_id in run_ids:
        content = redis_client.hget(
            _get_user_run_id_content_key(user_id), run_id.decode("utf-8")
        )
        result.append(
            RunHistoryItem(
                run_id=run_id.decode("utf-8"),
                user_id=user_id,
                content=content.decode("utf-8"),
            )
        )

    return result


def last_messages(
    user_id: str, run_id: str, agent_id: str = ""
) -> List[Dict[str, Any]]:
    result = {"results": []}
    try:
        session_scope = _build_session_scope(
            filters={
                "run_id": run_id,
                "user_id": user_id,
            }
        )

        last_messages = memory.db.get_last_messages(session_scope=session_scope)

        list = []
        for message in last_messages:
            list.append({"memory": message.get("content", "")})

        result["results"] = list
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")

    return result
