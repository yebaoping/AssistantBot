from dataclasses import dataclass
import logging
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from .agents import deep_agent
from .agents import react_agent
from .checkpointer import checkpointer
from .memory import add_user_run_id, last_messages, memory
from .models import ollama_model
from .prompts import ROUTE_PROMPT


logger = logging.getLogger(__name__)

MESSAGES_LIMIT = 2


@dataclass
class GraphContext:
    user_id: str
    thread_id: str


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summarized: bool = False


def _last_message_content(state: GraphState) -> str:
    message = ""

    if state["messages"] and state["messages"][-1]:
        if isinstance(state["messages"][-1], HumanMessage):
            message = state["messages"][-1].content
        elif isinstance(state["messages"][-1], dict):
            message = state["messages"][-1]["content"]

    return message


def retrieve_memory(
    state: GraphState, config: RunnableConfig, runtime: Runtime[GraphContext]
):
    user_id = runtime.context.get("user_id", "")
    thread_id = runtime.execution_info.thread_id

    add_user_run_id(user_id, thread_id, _last_message_content(state))

    if len(state["messages"]) == 1 or state.get("summarized", False):
        filters = {
            "user_id": user_id,
            "run_id": thread_id,
        }

        last_message = state["messages"][-1]
        message_content = ""
        if isinstance(last_message, HumanMessage):
            message_content = last_message.content
        elif isinstance(last_message, dict):
            message_content = last_message["content"]

        # 检索记忆
        # TODO 现有的嵌入模型不支持图片和音频，先直接返回前20条memory
        is_multimodal = False
        if isinstance(message_content, list) and len(message_content) > 0:
            for content in message_content:
                if isinstance(content, dict) and content.get("type", "").lower() in [
                    "image",
                    "audio",
                    "image_url",
                    "audio_url",
                ]:
                    is_multimodal = True
                    break

        memories = (
            memory.search(query=message_content, filters=filters, rerank=True)
            if not is_multimodal
            else memory.get_all(filters=filters)
        )

        # mem0的记忆依靠llm来生成，所以需要先检索memory，如果memory为空，则检索20条聊天记录作为记忆
        if not memories or not memories["results"]:
            memories = last_messages(user_id, thread_id)

        if memories and memories["results"]:
            state["messages"].insert(
                0,
                SystemMessage(
                    content="之前的对话记录:\n"
                    + "\n".join([mem["memory"] for mem in memories["results"]])
                ),
            )

    return {"messages": Overwrite(state["messages"])}


def route(state: GraphState):
    message = ""
    if state["messages"] and state["messages"][-1]:
        if isinstance(state["messages"][-1], HumanMessage):
            message = state["messages"][-1].content
        elif isinstance(state["messages"][-1], dict):
            message = state["messages"][-1]["content"]

    if not message:
        logger.warning(f"No message found, returning END. Original state: {state}")
        return END

    prompt = ROUTE_PROMPT.format(user_input=message)
    response = ollama_model.invoke(prompt)

    return "deep_agent" if "deep_agent" in response.content.lower() else "react_agent"


def deep_agent_invoke(
    state: GraphState, config: RunnableConfig, runtime: Runtime[GraphContext]
):
    response = deep_agent.invoke(
        {"messages": state["messages"]}, config=config, context=runtime.context
    )
    return {"messages": [response.get("messages", [])[-1]]}


def react_agent_invoke(
    state: GraphState, config: RunnableConfig, runtime: Runtime[GraphContext]
):
    response = react_agent.invoke(
        {"messages": state["messages"]}, config=config, context=runtime.context
    )
    return {"messages": [response.get("messages", [])[-1]]}


def summarize(
    state: GraphState, config: RunnableConfig, runtime: Runtime[GraphContext]
):
    if len(state["messages"]) > MESSAGES_LIMIT * 2:
        memories = []
        for message in state["messages"][:MESSAGES_LIMIT]:
            if isinstance(message, HumanMessage) and message.content:
                memories.append({"role": "user", "content": message.content})
            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and message.content
            ):
                memories.append({"role": "assistant", "content": message.content})

        if memories:
            memory.add(
                memories,
                user_id=runtime.context.get("user_id", ""),
                run_id=runtime.execution_info.thread_id,
            )

        messages = state["messages"][-MESSAGES_LIMIT * 2 :]
        return {
            "messages": Overwrite(messages),
            "summarized": True,
        }
    else:
        return {"messages": Overwrite(state["messages"])}


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("route", route)
    builder.add_node("deep_agent", deep_agent_invoke)
    builder.add_node("react_agent", react_agent_invoke)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "retrieve_memory")

    builder.add_conditional_edges(
        "retrieve_memory",
        route,
        {
            "deep_agent": "deep_agent",
            "react_agent": "react_agent",
            END: END,
        },
    )
    builder.add_edge("deep_agent", "summarize")
    builder.add_edge("react_agent", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)
