import uuid

from langchain.agents import create_agent

from agent.middlewares import MemoryMiddleware
from agent.models import ollama_model
from utils.image import encode_image


# pytest test_middlewares.py::test_MemoryMiddleware -s
def test_MemoryMiddleware():
    agent = create_agent(
        model=ollama_model,
        system_prompt="你是一个人工智能小助手，可以帮助用户解决各种问题。",
        middleware=[MemoryMiddleware("test_agent")],
    )

    thread_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    print(f"thread_id: {thread_id}")
    print(f"user_id: {user_id}")

    config = {"configurable": {"thread_id": thread_id}}
    context = {"user_id": user_id}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "第一个登陆月球的人是谁？"}]},
        config=config,
        context=context,
    )
    print("response: ", response)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "他获得过哪些荣誉？"}]},
        config=config,
        context=context,
    )
    print("response: ", response)


# pytest test_middlewares.py::test_MemoryMiddleware_Multimodal -s
def test_MemoryMiddleware_Multimodal():
    agent = create_agent(
        model=ollama_model,
        system_prompt="你是一个人工智能小助手，可以帮助用户解决各种问题。",
        middleware=[MemoryMiddleware("test_agent")],
    )

    thread_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    print(f"thread_id: {thread_id}")
    print(f"user_id: {user_id}")

    config = {"configurable": {"thread_id": thread_id}}
    context = {"user_id": user_id}

    base64_image = encode_image(
        "/Users/yebaoping/code/python/AssistantBot/resources/cat.jpg"
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片里有什么？"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]
        },
        config=config,
        context=context,
    )

    print("response: ", response)
