from agent.checkpointer import checkpointer
import uuid
from datetime import datetime


# pytest test_checkpointer.py::test_checkpointer -s
def test_checkpointer():
    thread_id = str(uuid.uuid4())
    print(f"thread_id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    checkpoint = checkpointer.get(config)
    assert checkpoint is None

    checkpoint_id = str(uuid.uuid4())
    ts = datetime.now().isoformat()
    config["configurable"]["checkpoint_id"] = checkpoint_id
    config["configurable"]["checkpoint_ns"] = "test_checkpointer"
    new_config = checkpointer.put(
        config=config,
        checkpoint={
            "v": 1,
            "id": checkpoint_id,
            "ts": ts,
            "channel_values": {
                "user_id": "user_1",
            },
            "channel_versions": {
                "user_id": "1.0",
            },
            "versions_seen": {
                "node_1": {
                    "user_id": "1.0",
                },
            },
            "updated_channels": [],
        },
        metadata={"source": "input", "step": 0, "parents": {}, "run_id": "run_1"},
        new_versions=None,
    )
    print(f"new_config: {new_config}")
    assert (
        new_config is not None
        and new_config["configurable"] is not None
        and new_config["configurable"]["thread_id"] == thread_id
        and new_config["configurable"]["checkpoint_ns"] == "test_checkpointer"
        and new_config["configurable"]["checkpoint_id"] == checkpoint_id
    )

    checkpoint = checkpointer.get(new_config)
    print(f"checkpoint: {checkpoint}")
    assert (
        checkpoint is not None
        and checkpoint["v"] == 1
        and checkpoint["id"] == checkpoint_id
        and checkpoint["ts"] == ts
        and checkpoint["channel_values"] == {"user_id": "user_1"}
        and checkpoint["channel_versions"] == {"user_id": "1.0"}
        and checkpoint["versions_seen"] == {"node_1": {"user_id": "1.0"}}
        and checkpoint["updated_channels"] == []
    )
