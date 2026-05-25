from __future__ import annotations
import base64
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import uuid

from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st

from agent.graph import build_graph
from agent.langfuse import langfuse_handler
from agent.memory import run_history
from utils.audio import transcribe


logging.basicConfig(level=logging.INFO)


@st.cache_resource
def get_graph():
    return build_graph()


def _get_thread_id() -> str:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id


def _get_chat_history(run_id: str) -> List[Dict[str, Any]]:
    result = []
    graph = get_graph()
    state = graph.get_state(config={"configurable": {"thread_id": run_id}})
    messages = state.values.get("messages", [])
    for message in messages:
        if isinstance(message, HumanMessage):
            result.append({"role": "user", "text": message.content})
        elif isinstance(message, AIMessage):
            result.append({"role": "assistant", "text": message.content})
    return result


def transcribe_audio(audio_file) -> str:
    suffix = Path(getattr(audio_file, "name", "audio.wav")).suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(audio_file.read())
            return transcribe(f.name)
    except Exception as e:
        st.error(f"语音转录失败：{e}")
        return ""


def build_human_message(
    text: str, img_bytes: bytes | None, img_mime: str | None
) -> HumanMessage:
    if not img_bytes:
        return HumanMessage(content=text)
    content = []
    if text:
        content.append({"type": "text", "text": text})
    data = base64.b64encode(img_bytes).decode()
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{data}"}}
    )
    return HumanMessage(content=content)


def send_message(
    user_id: str, text: str, img_bytes: bytes | None, img_mime: str | None
) -> None:
    human_msg = build_human_message(text, img_bytes, img_mime)

    with st.chat_message("user"):
        if img_bytes:
            st.image(img_bytes)
        if text:
            st.markdown(text)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            result = get_graph().invoke(
                {"messages": [human_msg]},
                config={
                    "configurable": {"thread_id": _get_thread_id()},
                    "callbacks": [langfuse_handler],
                },
                context={"user_id": user_id},
            )
        response = ""
        for msg in reversed(result["messages"]):
            if getattr(msg, "type", None) == "ai":
                c = msg.content
                if isinstance(c, str) and c.strip():
                    response = c
                    break
        st.markdown(response)

    st.session_state.chat_history.append(
        {"role": "user", "text": text, "image_bytes": img_bytes}
    )
    st.session_state.chat_history.append({"role": "assistant", "text": response})


st.set_page_config(
    page_title="AssistantBot — 你的人工智能小助理",
    page_icon="🤖",
    layout="centered",
)

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "request_processing" not in st.session_state:
    st.session_state.request_processing = False


@st.dialog("👋 欢迎使用 AssistantBot")
def _user_id_dialog() -> None:
    st.markdown("请输入您的 User ID 以开始对话。")
    uid = st.text_input("User ID", placeholder="例如：alice")
    if st.button("开始对话", disabled=not uid.strip(), use_container_width=True):
        st.session_state.user_id = uid.strip()
        st.rerun()


if not st.session_state.user_id:
    _user_id_dialog()
    st.stop()

user_id: str = st.session_state.user_id

if "run_history" not in st.session_state:
    st.session_state.run_history = run_history(user_id=user_id)

if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
    st.session_state.chat_history = _get_chat_history(run_id=_get_thread_id())


with st.sidebar:
    st.title("💬 历史对话")
    for msg in st.session_state.run_history:
        label = msg.content[0:10]
        try:
            result = json.loads(msg.content)
            if isinstance(result, list):
                for item in result:
                    if item.get("type", "") == "text":
                        label = item.get("text", "")[0:10]
        except Exception:
            pass

        if st.button(label + "...", key=f"msg_{msg.run_id}", use_container_width=True):
            st.session_state.thread_id = msg.run_id
            st.rerun()
    # st.divider()

st.title("🤖 AssistantBot — 你的人工智能小助理")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"])
        if msg.get("text"):
            if isinstance(msg["text"], list):
                for item in msg["text"]:
                    if item.get("type") == "image_url":
                        st.image(item.get("image_url").get("url"))
                    elif item.get("type") == "text":
                        st.markdown(item.get("text"))
            else:
                st.markdown(msg["text"])

prompt = st.chat_input(
    "输入消息、上传图片或录音…",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
    accept_audio=True,
    disabled=st.session_state.request_processing,
)

if prompt:
    text = prompt.text or ""

    img_bytes, img_mime = None, None
    if prompt.files:
        img_file = prompt.files[0]
        img_bytes = img_file.read()
        img_mime = getattr(img_file, "type", "image/jpeg")

    # langchain的ollama模型不支持语音转录，需要先转换成文本
    if prompt.audio:
        with st.spinner("转录中…"):
            transcribed = transcribe_audio(prompt.audio)
        text = f"{text} {transcribed}".strip() if text else transcribed

    if text or img_bytes:
        st.session_state.request_processing = True
        send_message(user_id, text, img_bytes, img_mime)
        st.session_state.request_processing = False
        st.rerun()

# streamlit run app_streamlit.py --server.port 8000
