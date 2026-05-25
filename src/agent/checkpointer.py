import os

from dotenv import load_dotenv
from langgraph.checkpoint.redis import RedisSaver

load_dotenv()

PREFIX = "assistant_bot"

# LangGraph 在 Redis 里用 RediSearch 维护 checkpoint 索引；首次使用前必须 setup，
# 否则会报：ResponseError: No such index <checkpoint_prefix>
checkpointer = RedisSaver(
    redis_url=os.getenv("REDIS_URL"),
    checkpoint_prefix=PREFIX,
    checkpoint_write_prefix=PREFIX,
)
checkpointer.setup()
