"""
FastAPI server — OpenAI-compatible chat API backed by LangChain + LangGraph.

Endpoints
---------
POST /v1/chat/completions   streaming (SSE) and non-streaming, with client-cancel support
POST /v1/audio/transcriptions   multipart audio → text via local Whisper

Auth
----
Set API_KEY env var.  Clients send: Authorization: Bearer <API_KEY>
If API_KEY is empty the server runs open (useful for local dev).

Run
---
uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.graph import build_graph
from agent.langfuse import langfuse_handler
from utils.audio import transcribe


load_dotenv()

app = FastAPI(title="AI Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────

_graph = None
_graph_lock = threading.Lock()


def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_graph()
    return _graph


# ── Auth ──────────────────────────────────────────────────────────────────────


def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        return
    if not authorization or authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Request / response models (OpenAI-compatible) ────────────────────────────


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatRequest(BaseModel):
    model: str = "agent"
    messages: list[ChatMessage]
    stream: bool = False
    user: str = "default"  # used as user_id / thread_id


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_human_message(msg: ChatMessage) -> HumanMessage:
    content = msg.content
    if isinstance(content, str):
        return HumanMessage(content=content)
    # Already a list of content parts (multimodal)
    return HumanMessage(content=content)


def _sse_chunk(content: str, cid: str, model: str, created: int) -> str:
    data = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_done(cid: str, model: str, created: int) -> str:
    data = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\ndata: [DONE]\n\n"


def _is_agent_text(chunk, metadata: dict) -> bool:
    return (
        metadata.get("langgraph_node") == "agent"
        and isinstance(getattr(chunk, "content", None), str)
        and bool(chunk.content)
    )


# ── Chat completions ──────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatRequest,
    _: None = Depends(verify_api_key),
):
    last_user = next((m for m in reversed(body.messages) if m.role == "user"), None)
    if not last_user:
        raise HTTPException(400, "No user message found")

    human_msg = _build_human_message(last_user)
    user_id = body.user
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    config = {
        "configurable": {"thread_id": user_id},
        "callbacks": [langfuse_handler],
    }

    # ── Non-streaming ─────────────────────────────────────────────────────────
    if not body.stream:
        result = get_graph().invoke(
            {"messages": [human_msg], "user_id": user_id, "memory_context": ""},
            config=config,
        )
        text = ""
        for m in reversed(result["messages"]):
            if getattr(m, "type", None) == "ai":
                c = m.content
                if isinstance(c, str) and c.strip():
                    text = c
                    break
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ── Streaming via SSE ─────────────────────────────────────────────────────
    # Run the synchronous LangGraph stream in a background thread.
    # A threading.Event propagates client-cancel into the thread.
    cancel = threading.Event()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=128)
    loop = asyncio.get_event_loop()

    def run_graph() -> None:
        try:
            for chunk, metadata in get_graph().stream(
                {"messages": [human_msg], "user_id": user_id, "memory_context": ""},
                config=config,
                stream_mode="messages",
            ):
                if cancel.is_set():
                    break
                if _is_agent_text(chunk, metadata):
                    asyncio.run_coroutine_threadsafe(
                        queue.put(_sse_chunk(chunk.content, cid, model_name, created)),
                        loop,
                    )
        except Exception as e:
            err = f"data: {json.dumps({'error': str(e)})}\n\n"
            asyncio.run_coroutine_threadsafe(queue.put(err), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    threading.Thread(target=run_graph, daemon=True).start()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    cancel.set()
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    break
                yield item
            yield _sse_done(cid, model_name, created)
        except asyncio.CancelledError:
            cancel.set()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Audio transcription ───────────────────────────────────────────────────────


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path: str | None = None
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name
        result = transcribe(tmp_path)
        return {"text": result["text"].strip()}
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
