import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.agents import coding_agent, coding_deep_agent, deep_agent, react_agent
from agent.daytona import SANDBOX_HOME_PATH, create_sandbox

load_dotenv()


# pytest test_agents.py::test_deep_agent -s
def test_deep_agent():
    thread_id = "test_thread_id"
    sandbox = create_sandbox(thread_id)

    path = os.path.join(SANDBOX_HOME_PATH, "upload", "zte-report-simple.pdf")

    with open(
        "../resources/zte-report-simple.pdf",
        "rb",
    ) as f:
        pdf_bytes = f.read()
        sandbox.fs.upload_file(
            pdf_bytes,
            path,
        )

    response = deep_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"读取年度财报文件{path}，对内容进行简要总结。"
                    # content="What is langgraph?"
                )
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    print(response)


# pytest test_agents.py::test_react_agent -s
def test_react_agent():
    response = react_agent.invoke(
        {"messages": [{"role": "user", "content": "请自我介绍一下"}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context={"user_id": str(uuid.uuid4())},
    )
    print(response)


# pytest test_agents.py::test_react_agent_generate_image -s
def test_react_agent_generate_image():
    response = react_agent.invoke(
        {"messages": [{"role": "user", "content": "生成一张长发蓝眼睛的漂亮女孩图片。"}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    assert response.get("messages", [])[-1] is not None
    with open("test_react_agent_generate_image.md", "wt") as f:
        print(response.get("messages", [])[-1].content,file=f)


# pytest test_agents.py::test_coding_agent -s
def test_coding_agent():
    response = coding_agent.invoke(
        {"messages": [{"role": "user", "content": "使用代码计算10的阶乘？"}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context={"user_id": str(uuid.uuid4())},
    )
    print(response)


# pytest test_agents.py::test_coding_deep_agent -s
def test_coding_deep_agent():
    response = coding_deep_agent.invoke(
        {"messages": [{"role": "user", "content": "使用代码计算10的阶乘？"}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context={"user_id": str(uuid.uuid4())},
    )
    print(response)
