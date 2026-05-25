import asyncio

from langchain.tools import ToolRuntime

from agent.tools.execute import code_execute, command_execute, execute
from agent.tools.text_to_image import text_to_image
from agent.tools.web_search import WebSearch, simple_web_search, web_search


# pytest test_tools.py::test_simple_web_search -s
def test_simple_web_search():
    print(simple_web_search.name)
    print(simple_web_search.description)

    response = simple_web_search.invoke("杭州今天天气")
    print(response)

    response = simple_web_search.invoke("西溪湿地票价")
    print(response)


# pytest test_tools.py::test_web_search -s
def test_web_search():
    print(web_search.name)
    print(web_search.description)

    response = web_search.invoke("杭州今天天气")
    print(response)

    response = web_search.invoke("西溪湿地票价")
    print(response)


# pytest test_tools.py::test_WebSearch -s
def test_WebSearch():
    web_search = WebSearch(
        max_results=1, search_depth="basic", include_raw_content=False
    )
    response = web_search.invoke("杭州今天天气")
    print(response)


class MockContext:
    pass


# pytest test_tools.py::test_execute_code -s
def test_execute_code():
    thread_id = "test_execute_code"

    command = """python -c 'import math\nresult = math.factorial(10)\nprint(f"10的阶乘 = {result}")\n\n# 同时展示计算过程\nprocess = " × ".join(str(i) for i in range(1, 11))\nprint(f"计算过程: {process} = {result}")'"""
    runtime = ToolRuntime(
        context=MockContext(),
        state={},
        config={"configurable": {"thread_id": thread_id}},
        stream_writer=lambda x: None,
        tool_call_id="mock_tool_call_001",
        store=None,
    )

    response = execute.invoke({"command": command, "runtime": runtime})
    print(response)


# pytest test_tools.py::test_execute_command -s
def test_execute_command():
    thread_id = "test_execute_command"

    command = "python --version"
    runtime = ToolRuntime(
        context=MockContext(),
        state={},
        config={"configurable": {"thread_id": thread_id}},
        stream_writer=lambda x: None,
        tool_call_id="mock_tool_call_001",
        store=None,
    )

    response = execute.invoke({"command": command, "runtime": runtime})
    print(response)


# pytest test_tools.py::test_code_execute -s
def test_code_execute():
    thread_id = "test_code_execute"

    code = 'import math\nresult = math.factorial(10)\nprint(f"10的阶乘 = {result}")\n\n# 同时展示计算过程\nprocess = " × ".join(str(i) for i in range(1, 11))\nprint(f"计算过程: {process} = {result}")'
    runtime = ToolRuntime(
        context=MockContext(),
        state={},
        config={"configurable": {"thread_id": thread_id}},
        stream_writer=lambda x: None,
        tool_call_id="mock_tool_call_001",
        store=None,
    )

    response = code_execute.invoke({"code": code, "runtime": runtime})
    print(response)


# pytest test_tools.py::test_command_execute -s
def test_command_execute():
    thread_id = "test_command_execute"

    command = """python -c 'import math\nresult = math.factorial(10)\nprint(f"10的阶乘 = {result}")\n\n# 同时展示计算过程\nprocess = " × ".join(str(i) for i in range(1, 11))\nprint(f"计算过程: {process} = {result}")'"""
    runtime = ToolRuntime(
        context=MockContext(),
        state={},
        config={"configurable": {"thread_id": thread_id}},
        stream_writer=lambda x: None,
        tool_call_id="mock_tool_call_001",
        store=None,
    )

    response = command_execute.invoke({"command": command, "runtime": runtime})
    print(response)


# pytest test_tools.py::test_text_to_image -s
def test_text_to_image():
    tool = text_to_image()
    response = asyncio.run(
        tool.ainvoke({"prompt": "A beautiful girl with long hair and blue eyes"})
    )
    assert response is not None
    assert response[-1]["base64"] is not None

    import base64

    binary_data = base64.b64decode(response[-1]["base64"])
    with open("test_text_to_image.png", "wb") as f:
        f.write(binary_data)
