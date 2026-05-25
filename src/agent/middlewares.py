from typing import Any, Callable
from typing import Optional

from dotenv import load_dotenv
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    after_agent,
    dynamic_prompt,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from .memory import memory

load_dotenv()


class MemoryMiddleware(AgentMiddleware):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory = memory

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        # print("messages: ", request.messages)
        # print("user_id: ", request.runtime.context.get("user_id","None"))
        # print("checkpoint_id: ", request.runtime.execution_info.checkpoint_id)
        # print("task_id: ", request.runtime.execution_info.task_id)
        # print("thread_id: ", request.runtime.execution_info.thread_id)
        # print("run_id: ", request.runtime.execution_info.run_id)

        message = request.messages[-1]

        user_id = request.runtime.context.get("user_id", "")
        thread_id = request.runtime.execution_info.thread_id
        filters = {
            "user_id": user_id,
            "run_id": thread_id,
            # "agent_id": self.agent_id
        }

        # 检索记忆
        # TODO 现有的嵌入模型不支持图片和音频，先直接返回前20条memory
        is_multimodal = False
        if isinstance(message.content, list) and len(message.content) > 0:
            for content in message.content:
                if isinstance(content, dict) and content.get("type", "").lower() in [
                    "image",
                    "audio",
                    "image_url",
                    "audio_url",
                ]:
                    is_multimodal = True
                    break

        memory = (
            self.memory.search(query=message.content, filters=filters, rerank=True)
            if not is_multimodal
            else self.memory.get_all(filters=filters)
        )
        if memory and memory["results"]:
            request.messages.insert(
                0,
                SystemMessage(
                    content="之前的对话记录:"
                    + "\n".join([mem["memory"] for mem in memory["results"]])
                ),
            )

        memories = []

        # 工具调用不保存到记忆中
        if isinstance(message, HumanMessage):
            memories.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage) and not (
            message.tool_call_id or message.tool_calls
        ):
            memories.append({"role": "assistant", "content": message.content})

        response = handler(request)

        # 工具调用不保存到记忆中
        if isinstance(response, ModelResponse):
            for message in response.result:
                if not isinstance(message, ToolMessage) and not (
                    isinstance(message, AIMessage) and message.tool_calls
                ):
                    memories.append({"role": "assistant", "content": message.content})
        elif isinstance(response, AIMessage) and not response.tool_calls:
            memories.append({"role": "assistant", "content": response.content})

        if len(memories) > 0:
            self.memory.add(
                memories,
                user_id=user_id,
                run_id=thread_id,
                # agent_id=self.agent_id,
            )

        return response


@dynamic_prompt
def retrieve_memory_prompt(request: ModelRequest) -> Optional[str]:
    print(request.system_prompt)
    return request.system_prompt


@after_agent
def add_agent_memory(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state.get("messages", [])

    user_id = runtime.context.get("user_id", "")
    thread_id = runtime.execution_info.thread_id

    memories = []
    for message in messages:
        if isinstance(message, HumanMessage) and message.content:
            memories.append({"role": "user", "content": message.content})
        elif (
            isinstance(message, AIMessage)
            and not message.tool_calls
            and message.content
        ):
            memories.append({"role": "assistant", "content": message.content})

    memory.add(
        memories[-2:],
        user_id=user_id,
        run_id=thread_id,
        # agent_id=self.agent_id,
    )

    return None


@after_agent
def add_image_to_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    images = []
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and message.name == "text_to_image"
            and isinstance(message.content, list)
            and message.content
        ):
            for c in message.content:
                if c.get("type", "") == "image" and c.get("base64", ""):
                    images.append(c.get("base64", ""))

    if isinstance(messages[-1], AIMessage) and images:
        messages[-1].content = (
            "\n\n".join([f"![图片描述](data:image/png;base64,{img})" for img in images])
            + "\n\n" + messages[-1].content
        )

    return None
