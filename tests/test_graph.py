import uuid

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from utils.image import encode_image


# pytest test_graph.py::test_graph -s
def test_graph():
    graph = build_graph()

    thread_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    messages = [HumanMessage(content="")]
    # messages = [{"role": "user", "content": "请自我介绍一下"}]

    response = graph.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
        context={"user_id": user_id},
    )
    print(response)


def test_graph_image_message():
    graph = build_graph()

    thread_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    base64_image = encode_image(
        "/Users/yebaoping/code/python/AssistantBot/resources/cat.jpg"
    )

    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "这张图片里有什么？"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
    ]

    response = graph.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
        context={"user_id": user_id},
    )
    print(response)
