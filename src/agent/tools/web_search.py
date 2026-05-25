import re
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_tavily import TavilySearch


load_dotenv()


class WebSearch(TavilySearch):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[
            Literal["basic", "advanced", "fast", "ultra-fast"]
        ] = None,
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        response = super()._run(
            query,
            include_domains,
            exclude_domains,
            search_depth,
            include_images,
            time_range,
            topic,
            start_date,
            end_date,
            run_manager,
            **kwargs,
        )
        return _filter_results(response)

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[
            Literal["basic", "advanced", "fast", "ultra-fast"]
        ] = "basic",
        include_images: Optional[bool] = False,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = "general",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        response = await super()._arun(
            query,
            include_domains,
            exclude_domains,
            search_depth,
            include_images,
            time_range,
            topic,
            start_date,
            end_date,
            run_manager,
            **kwargs,
        )
        return _filter_results(response)


def _remove_images_and_links(content: str) -> str:
    # 1. 删除 Markdown 图片
    # 例如: ![alt](image.png)
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

    # 2. 删除 Markdown 链接，仅保留文字
    # 例如: [OpenAI](https://openai.com) -> OpenAI
    content = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", content)

    # 3. 删除 HTML img 标签
    content = re.sub(r"<img[^>]*>", "", content, flags=re.IGNORECASE)

    # 4. 删除裸链接
    # 例如: https://example.com
    content = re.sub(r"https?://\S+", "", content)

    return content.strip()


def _filter_results(response: Dict[str, Any]) -> Dict[str, Any]:
    if not response or not response.get("results", []):
        return response

    results = []
    for result in response.get("results", []):
        if result.get("score", 0) < 0.8 or not result.get("content", ""):
            continue

        content = _remove_images_and_links(result.get("content", ""))
        if content:
            results.append(content)

    if not results:
        return response

    return {"results": results}


simple_web_search = WebSearch(
    max_results=1,
    search_depth="basic",
    include_raw_content=False,
)

web_search = WebSearch()
