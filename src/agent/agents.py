from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain_quickjs import CodeInterpreterMiddleware

from agent.tools.browser import get_browser_tools

from .backend import sandbox_backend
from .checkpointer import checkpointer
from .daytona import SANDBOX_SKILLS_PATH
from .middlewares import add_image_to_message
from .models import anthropic_model, deepseek_model, ollama_model, openai_model
from .prompts import (
    BROWSER_AGENT_PROMPT,
    CODING_AGENT_PROMPT,
    CODING_PROMPT,
    COMMAND_EXECUTE_PROMPT,
    REACT_AGENT_PROMPT,
    WEB_SEARCH_AGENT_PROMPT,
)
from .tools.execute import code_execute, command_execute
from .tools.text_to_image import text_to_image
from .tools.web_search import simple_web_search, web_search

web_search_agent = {
    "name": "web_search",
    "description": "从互联网上检索信息并汇总结果的智能体。适用于需要实时网页搜索的任务。",
    "system_prompt": WEB_SEARCH_AGENT_PROMPT,
    "tools": [web_search],
    "model": ollama_model,
}


coding_agent = create_agent(
    name="coding",
    model=anthropic_model,
    tools=[code_execute],
    # tools=[execute],
    system_prompt=CODING_PROMPT,
    checkpointer=checkpointer,
    debug=True,
)

command_execute_agent = create_agent(
    name="command_execute",
    model=deepseek_model,
    tools=[command_execute],
    # tools=[execute],
    system_prompt=COMMAND_EXECUTE_PROMPT,
    checkpointer=checkpointer,
    debug=True,
)

coding_deep_agent = create_deep_agent(
    name="coding",
    model=anthropic_model,
    system_prompt=CODING_AGENT_PROMPT,
    checkpointer=checkpointer,
    backend=sandbox_backend,
)

browser_agent = create_agent(
    name="browser",
    model=openai_model,
    tools=get_browser_tools(),
    system_prompt=BROWSER_AGENT_PROMPT,
    checkpointer=checkpointer,
)

# TODO 通过create_agent创建普通agent，通过SubAgentMiddleware创建使用子agent来替换create_deep_agent创建的deep_agent
# 主Agent使用gpt，负责思考、规划，总结。子Agent使用开源模型，负责执行任务。
deep_agent = create_deep_agent(
    model=deepseek_model,
    # system_prompt=DEEP_AGENT_PROMPT,
    skills=[SANDBOX_SKILLS_PATH],
    backend=sandbox_backend,
    middleware=[
        add_image_to_message,
        CodeInterpreterMiddleware(skills_backend=sandbox_backend),
    ],
    checkpointer=checkpointer,
    tools=[text_to_image()],
    interrupt_on={
        "write_file": False,
        "edit_file": False,
        "delete_file": True,
        "read_file": False,
        "execute_file": False,
        "list_files": False,
        "create_file": False,
        "rename_file": True,
        "move_file": True,
        "copy_file": False,
    },
    subagents=[
        web_search_agent,
        CompiledSubAgent(
            name="coding",
            description="具有沙盒访问权限的编程智能体。可以编写和执行代码来解决问题，适用于需要编程的任务。",
            runnable=coding_agent,
        ),
        CompiledSubAgent(
            name="browser",
            description="擅长使用和操作浏览器的浏览器操作智能体。可以使用浏览器访问和浏览网页，操作网页。",
            runnable=browser_agent,
        ),
        CompiledSubAgent(
            name="command_execute",
            description="执行shell命令的命令执行智能体。可以执行shell命令，并返回结果。",
            runnable=command_execute_agent,
        ),
    ],
    debug=True,
)


react_agent = create_agent(
    model=ollama_model,
    tools=[simple_web_search, text_to_image()],
    system_prompt=REACT_AGENT_PROMPT,
    middleware=[add_image_to_message],
    checkpointer=checkpointer,
)
