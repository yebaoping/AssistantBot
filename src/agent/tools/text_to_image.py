import asyncio
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


def _with_sync_support(tool: StructuredTool) -> StructuredTool:
    coroutine = tool.coroutine

    def func(**kwargs):
        return asyncio.run(coroutine(**kwargs))

    return tool.model_copy(update={"func": func})


async def _load_mcp_tool() -> StructuredTool:
    client = MultiServerMCPClient(
        {
            "StableDiffusion": {
                "transport": "streamable_http",
                "url": os.getenv("STABLE_DIFFUSION_MCP_URL"),
            },
        }
    )
    return (await client.get_tools())[0]


@lru_cache(maxsize=1)
def text_to_image() -> StructuredTool:
    tool = asyncio.run(_load_mcp_tool())
    return _with_sync_support(tool)
