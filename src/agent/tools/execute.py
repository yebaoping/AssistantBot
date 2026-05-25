from langchain.tools import ToolRuntime, tool

from ..backend import sandbox_backend
from ..daytona import create_sandbox


@tool
def execute(command: str, runtime: ToolRuntime) -> str:
    """
    在sandbox中执行shell命令或者代码，并返回结果。

    Args:
        command: shell命令，如果要执行python代码必须使用"python -c"命令 。
    """

    try:
        result = sandbox_backend.execute(command)
        if result.exit_code != 0:
            return f"执行失败，错误信息：{result.output.strip().strip('<stderr>').strip('</stderr>')}"

        return result.output.strip().strip("<stdout>").strip("</stdout>")
    except Exception as e:
        return str(e)


@tool
def code_execute(code: str, runtime: ToolRuntime) -> str:
    """
    在sandbox中运行python代码，并返回结果。

    Args:
        code: python代码
    """

    try:
        sandbox = create_sandbox(runtime.config["configurable"]["thread_id"])

        result = sandbox.process.code_run(code)

        if result.exit_code != 0:
            return f"代码执行失败，错误信息：{result.result}"

        return result.result
    except Exception as e:
        return str(e)


@tool
def command_execute(command: str, runtime: ToolRuntime) -> str:
    """
    在sandbox中运行shell命令，并返回结果。

    Args:
        command: shell命令
    """

    try:
        sandbox = create_sandbox(runtime.config["configurable"]["thread_id"])

        result = sandbox.process.exec(command)

        if result.exit_code != 0:
            return f"命令执行失败，错误信息：{result.result}"

        return result.result
    except Exception as e:
        return str(e)
