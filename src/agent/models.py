import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic



load_dotenv()

openai_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    reasoning={
        "effort": "medium",  # Can be "low", "medium", or "high"
        "summary": "auto",  # Can be "auto", "concise", or "detailed"
    },
)

ollama_model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_HOST"),
    reasoning=True,
)

deepseek_model = ChatDeepSeek(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

anthropic_model = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL"),
)